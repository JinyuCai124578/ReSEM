import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.LISA import LISAForCausalLM
from model.LISA_qwen import LISAQwenForCausalLM
from model.LISA_qwsa import QWSAForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="model/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./lisa_model", type=str, required=True)
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument('--train_mask_decoder_only', action='store_true', default=False)
    parser.add_argument('--full_finetune', action='store_true', default=False)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = "[PAD]"  # 设置默认填充标记

    # 确保 pad_token 在词汇表中
    if tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})

    # 设置 pad_token_id
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "vision_tower": args.vision_tower,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    if "Qwen2.5-VL" in args.version:
        model_args["vision_pretrained"] = args.vision_pretrained
        model=QWSAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=False, **model_args
    )
    elif "qwen" in args.version:
        model = LISAQwenForCausalLM.from_pretrained(
            args.version,
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
        )
    else:
        model = LISAForCausalLM.from_pretrained(
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if "Qwen2.5-VL" in args.version:
        pass
    else:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
        model.get_model().initialize_lisa_modules(model.get_model().config)

    lora_r = args.lora_r if not args.train_mask_decoder_only else 0
    if args.full_finetune:
        lora_r = 0
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    if args.train_mask_decoder_only:
        assert args.train_from_scratch == False, "train_from_scratch not supported when training mask decoder only"
    if not args.train_from_scratch:
        print("loading from pretrained")
        lisa_params=torch.load('model/lisa_params.pt', map_location='cuda')
        for name, param in lisa_params.items():
            # print(name)
            name="base_model.model."+name
            # pdb.set_trace()
            if name in model.state_dict():
                # print("load {}".format(name))
                model.state_dict()[name].copy_(param)
        del lisa_params
    
    # 将 meta tensor 移到设备并初始化（全零）
    for name, param in model.named_parameters():
        if param.device.type=='meta':
            new_param = torch.zeros_like(param, device='cuda')
            # to cuda
            module_path = name.split('.')
            target = model
            for part in module_path[:-1]:  # 遍历到最后一层前（父模块）
                target = getattr(target, part)
            
            # 设置最终参数
            setattr(target, module_path[-1], torch.nn.Parameter(new_param))
            print(f"Initialized meta tensor '{name}' to zeros on CUDA")

    # state_dict = torch.load(args.weight, map_location="cpu")
    # model.load_state_dict(state_dict, strict=True)

    # for deepspeed version higher than 0.15.2 https://github.com/dvlab-research/LISA/issues/90
    import json
    import gc
    index_file = os.path.join(args.weight, "pytorch_model.bin.index.json")

    with open(index_file, "r", encoding="utf-8") as f:
        index_dict = json.load(f)

    weight_map = index_dict["weight_map"]  # dict

    shard_files = set(weight_map.values()) 

    for shard_file in tqdm(shard_files, desc="Loading shards into model"):
        shard_path = os.path.join(args.weight, shard_file)

        shard_state = torch.load(shard_path, map_location="cpu", weights_only=True)

        # 加载到 model（只写入本 shard 内的参数）
        model.load_state_dict(shard_state, strict=False)

        # 释放内存
        del shard_state
        gc.collect()
        torch.cuda.empty_cache()

    """
    RuntimeError: Error(s) in loading state_dict for PeftModelForCausalLM:
            size mismatch for base_model.model.model.embed_tokens.weight: copying a param with shape torch.Size([32004, 5120]) from checkpoint, the shape in current model is torch.Size([32003, 5120]).
            size mismatch for base_model.model.lm_head.weight: copying a param with shape torch.Size([32004, 5120]) from checkpoint, the shape in current model is torch.Size([32003, 5120]).
    """

    if lora_r>0:
        model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    # warning
    # model.generation_config.validate()  # 手动调用验证，观察其他潜在问题（可选）
    # model.generation_config._validate = False  # 强制跳过验证
    model.generation_config.pad_token_id=0
    # warning
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
