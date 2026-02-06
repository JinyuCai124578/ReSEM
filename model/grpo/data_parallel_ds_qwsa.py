import torch.nn as nn
import torch
import copy
import deepspeed
from typing import Any

def selective_log_softmax(logits, input_ids):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    # import pdb; pdb.set_trace()
    input_ids = torch.clamp(input_ids, min=0)
    logits = model(input_ids=input_ids, attention_mask=attention_mask,super=True).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :].contiguous()
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, input_dict, num_generations=4, max_completion_length=32):
    prompt_ids = input_dict["prompt_ids"] # torch.Size([20, 163])
    prompt_mask = input_dict["attention_masks_prompts"] # torch.Size([20, 163])
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0) # torch.Size([240, 163])
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    # 先按input_dict['questions_list'] 复制
    gt_masks=input_dict["masks_list"]
    label_list=input_dict["label_list"]
    resize_list=input_dict["resize_list"]
    
    images= input_dict["images"] 
    images=images.repeat_interleave(num_generations, dim=0) # torch.Size([240, 3, 224, 224])
    pixel_values=input_dict["pixel_values"] 
    pixel_values=pixel_values.repeat_interleave(num_generations, dim=0) # torch.Size([240, 3, 224, 224])
    gt_masks_repeat=[gt_mask for gt_mask in gt_masks for _ in range(num_generations)]
    label_list_repeat=[label for label in label_list for _ in range(num_generations)]
    resize_list_repeat=[resize for resize in resize_list for _ in range(num_generations)]
    # import pdb; pdb.set_trace()
    outputs=model(
        images=images,
        input_ids=prompt_ids,
        attention_masks=prompt_mask,
        # label_list=label_list,
        # resize_list=resize_list,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_completion_length,
        temperature=1.0,
        output_hidden_states=True,
        return_dict_in_generate=True,
        do_sample=True,
        early_stopping=False,
        grpo=True,
    )
    output_ids=outputs['output_ids']
    prompt_length = prompt_ids.shape[1]
    completion_ids=output_ids[:, prompt_length:]
    completion_mask=create_completion_mask(completion_ids, tokenizer.eos_token_id)
    pred_low_res_masks=outputs['pred_low_res_masks']
    pred_masks=[]
    for i in range(len(pred_low_res_masks)):
        pred_low_res_mask=pred_low_res_masks[i:i+1]
        pred_mask=model.module.visual_model.postprocess_masks(
            pred_low_res_mask,
            input_size=resize_list_repeat[i],
            original_size=label_list_repeat[i].shape)
        pred_masks.append((pred_mask[0][0]>0).int())
    # import pdb; pdb.set_trace()
    return prompt_ids, prompt_mask, completion_ids, completion_mask, pred_masks, gt_masks_repeat


def generate_rollout_data(model, ref_model, tokenizer, input_dict, num_generations, max_completion_length):
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask, pred_masks, gt_masks = generate_completions(
            model, tokenizer, input_dict, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
        formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=False)}] for ids in completion_ids]

        answers = []
        for a in input_dict['classes_list']:
            answers+=a
        repeated_answers=[a for a in answers for _ in range(num_generations)]
        
        return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(input_dict['classes_list']),
        "num_generations": num_generations,
        "pred_masks": pred_masks,
        "gt_masks": gt_masks,

    }


def grpo_loss(model,rollout_data, reward_function, beta=0.01, epsilon=0.2):
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio=torch.exp(token_log_probs-old_log_probs)
    rewards = torch.tensor(
        reward_function(
            completions=rollout_data["formatted_completions"], 
            answers=rollout_data["repeated_answers"],
            pred_masks=rollout_data["pred_masks"],
            gt_masks=rollout_data["gt_masks"],
            ),
        dtype=torch.float32,
    ).cuda()
    # batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(-1, num_generations)
    avg_reward=rewards.mean().item()
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    # 计算PPO代理损失  防止策略更新过大
    surr1 = ratio * advantages #cuda:1 and cuda:0
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    # Loss = -E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) - β * KL(P_ref || P_policy) ]
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss, avg_reward


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


def train_with_grpo(model_engine, tokenizer, train_dataloader,train_iter, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, device_ids=None):
    # assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"
    # model = nn.DataParallel(model, device_ids=device_ids)
    # print(f"Model wrapped with DataParallel across GPUs: {device_ids}")
    # Outer loop: iterative GRPO updates.
    dtype = next(model_engine.module.parameters()).dtype
    mp_size=len(device_ids) if device_ids is not None else 1
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        # ------------------------------------------------
        # 构造 reference model —— 深拷贝训练模型并冻结
        # ------------------------------------------------
        ref_base_model = copy.deepcopy(model_engine.module).cpu().eval()
        for p in ref_base_model.parameters():
            p.requires_grad = False

        ref_infer_engine = deepspeed.init_inference(
            model=ref_base_model,
            mp_size=1,
            dtype=dtype,
            replace_with_kernel_inject=False     # ← 禁止 kernel 注入
        )
        
        # ------------------------------------------------
        # 构造 policy 推理模型 —— 参数与训练 engine 同步，但不参与梯度
        # ------------------------------------------------
        policy_base_model = copy.deepcopy(model_engine.module).cpu().eval()
        for p in policy_base_model.parameters():
            p.requires_grad = False

        policy_infer_engine = deepspeed.init_inference(
            model=policy_base_model,
            mp_size=1,
            dtype=dtype,
            replace_with_kernel_inject=False     # ← 禁止 kernel 注入
        )

        # ------------------------------------------------
        # 开始训练 iteration
        # ------------------------------------------------
        model_engine.train()

        # Inner loop: your original training steps.
        for step in range(num_steps):
            # batch_samples = random.sample(list(train_dataloader.dataset), batch_size)
            try:
                batch_samples = next(train_iter)
            except:
                train_iter = iter(train_dataloader)
                batch_samples = next(train_iter)
            
            batch_samples=dict_to_cuda(batch_samples)

            rollout_data = generate_rollout_data(
                policy_infer_engine,
                ref_infer_engine,
                tokenizer,
                batch_samples,
                num_generations,
                max_completion_length
            )

            for grpo_iter in range(mu):
                loss, avg_reward = grpo_loss(
                    model=model_engine,
                    rollout_data=rollout_data,
                    reward_function=reward_function,
                    beta=beta,
                    epsilon=epsilon
                )
                # optimizer.zero_grad()
                # import pdb; pdb.set_trace()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                # optimizer.step()
                model_engine.backward(loss)
                model_engine.step()
                print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                      f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item()}, avg_reward: {avg_reward}")
                # ------------------------------------------------
                # 训练一轮(?)结束时，同步 policy_infer_engine 的参数
                # ------------------------------------------------
                del rollout_data
                policy_base_model.load_state_dict(copy.deepcopy(model_engine.module.state_dict()))
                torch.cuda.empty_cache()
        del ref_infer_engine, policy_infer_engine, ref_base_model, policy_base_model
        torch.cuda.empty_cache()

    return model_engine

def recursive_detach(t: Any, to_cpu: bool = False) -> Any:
    """递归 detach/clone 所有 torch.Tensor，确保 requires_grad=False。
    对 dict/list/tuple 递归处理，返回同结构的数据（Tensor 都为 detached clones）。
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().clone()
        t.requires_grad_(False)
        return t.cpu() if to_cpu else t
    elif isinstance(t, dict):
        return {k: recursive_detach(v, to_cpu=to_cpu) for k, v in t.items()}
    elif isinstance(t, (list, tuple)):
        lst = [recursive_detach(x, to_cpu=to_cpu) for x in t]
        return type(t)(lst)
    else:
        return t

def train_with_grpo_epoch(ref_infer_engine,policy_infer_engine,policy_base_model,model_engine, tokenizer, train_dataloader,train_iter, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, device_ids=None):

    # dtype = next(model_engine.module.parameters()).dtype
    # for iteration in range(num_iterations):
    #     print(f"\nIteration {iteration+1}/{num_iterations}")

        # ------------------------------------------------
        # 构造 reference model —— 深拷贝训练模型并冻结
        # ------------------------------------------------
    # ref_base_model = copy.deepcopy(model_engine.module).cpu().eval()
    # for p in ref_base_model.parameters():
    #     p.requires_grad = False

    # ref_infer_engine = deepspeed.init_inference(
    #     model=ref_base_model,
    #     mp_size=1,
    #     dtype=dtype,
    #     replace_with_kernel_inject=False     # ← 禁止 kernel 注入
    # )
    
    # # ------------------------------------------------
    # # 构造 policy 推理模型 —— 参数与训练 engine 同步，但不参与梯度
    # # ------------------------------------------------
    # policy_base_model = copy.deepcopy(model_engine.module).cpu().eval()
    # for p in policy_base_model.parameters():
    #     p.requires_grad = False

    # policy_infer_engine = deepspeed.init_inference(
    #     model=policy_base_model,
    #     mp_size=1,
    #     dtype=dtype,
    #     replace_with_kernel_inject=False     # ← 禁止 kernel 注入
    # )
    # 构造 policy 推理模型 —— 不再 deepcopy，直接共享 model_engine.module 权重
    # policy_base_model = model_engine.module    # 直接引用，不拷贝
    # policy_base_model.eval()

    # # 参数已经在训练中由 model_engine 更新，因此推理模型会实时共享更新
    # for p in policy_base_model.parameters():
    #     p.requires_grad = False  # 只是确保推理时不产生梯度

    # policy_infer_engine = deepspeed.init_inference(
    #     model=policy_base_model,
    #     mp_size=1,
    #     dtype=dtype,
    #     replace_with_kernel_inject=False
    # )



    # ------------------------------------------------
    # 开始训练 iteration
    # ------------------------------------------------
    model_engine.train()

    # Inner loop: your original training steps.
    for step in range(num_steps):
        # batch_samples = random.sample(list(train_dataloader.dataset), batch_size)
        try:
            batch_samples = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            batch_samples = next(train_iter)
        
        batch_samples=dict_to_cuda(batch_samples)

        rollout_data = generate_rollout_data(
            policy_infer_engine,
            ref_infer_engine,
            tokenizer,
            batch_samples,
            num_generations,
            max_completion_length
        )
        # ---- 修复：把 inference tensor 全部 clone 成普通 tensor ----
        # rollout_data=recursive_detach(rollout_data, to_cpu=False)
        for grpo_iter in range(mu):
            loss, avg_reward = grpo_loss(
                model=model_engine,
                rollout_data=rollout_data,
                reward_function=reward_function,
                beta=beta,
                epsilon=epsilon
            )
            # optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            # optimizer.step()
            model_engine.backward(loss)
            model_engine.step()
            print(f"Step {step+1}/{num_steps}, "
                    f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item()}, avg_reward: {avg_reward}")
            # ------------------------------------------------
            # 训练一轮(?)结束时，同步 policy_infer_engine 的参数
            # ------------------------------------------------
            del rollout_data, loss
            policy_base_model.load_state_dict(copy.deepcopy(model_engine.module.state_dict()))
            torch.cuda.empty_cache()
    # del ref_infer_engine, policy_infer_engine, ref_base_model, policy_base_model
    torch.cuda.empty_cache()
    return model_engine
