import torch

def format_reward_cal(completions, **kwargs):
   """
   Assigns a reward for adhering to the desired XML format.
   whether [SEG] is in the completions
   """
   responses = [completion[0]['content'] for completion in completions]
   rewards = []
   format_scores = []
   for response in responses:
       score = 0.0
       if "[SEG]" in response: score += 0.2
       rewards.append(score)
       format_scores.append(score)
   return rewards

def correctness_reward_cal(completions, answers, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.
    answer class in completions: 1.0
    otherwise: 0.0
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for r, a in zip(responses, answers):
        a.replace("_", " ")
        if a.lower() in r.lower():  # Exact match case
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def iou_reward_cal(predictions, ground_truths, **kwargs):
    """
    Assigns a reward based on the Intersection over Union (IOU) metric.
    IOU = intersection / union
    """
    rewards = []
    for pred, gt in zip(predictions, ground_truths):
        intersection=torch.logical_and(pred, gt).sum()
        union=torch.logical_or(pred, gt).sum()
        iou=intersection/union
        rewards.append(iou)
    
    return rewards

def key_words_reward_cal(completions, answers, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    rewards = []

    # ----- Your original keyword lists -----
    spatial_keywords = [
        "boundary", "region", "area", "location", "position",
        "center", "left", "right", "top", "bottom", "upper", "lower"
    ]

    function_keywords = [
        "function", "role", "purpose", "responsible for", "enables",
        "allows", "involved in", "contributes to", "regulates", "essential for"
    ]

    # ----- New reasoning keywords -----
    reasoning_keywords = [
        "therefore", "because", "thus", "infer", "deduce",
        "based on", "suggests", "indicates", "reasoning", "conclusion"
    ]

    # ---- Reward weights (you can tune them) ----
    w_spatial_function = 1.0
    w_reasoning = 0.2           # small reward per reasoning keyword
    max_reasoning_reward = 1.0  # cap to avoid keyword spamming
    # w_spatial_function = 0
    # w_reasoning = 0           # small reward per reasoning keyword
    # max_reasoning_reward = 0  # cap to avoid keyword spamming
    w_seg_after_spatial = 1.5   # encourage SEG after spatial info

    for response in responses:
        r = response.lower()
        reward = 0.0

        # ---- 1. Reward for spatial or functional keywords ----
        has_spatial_keyword = any(k in r for k in spatial_keywords)
        has_function_keyword = any(k in r for k in function_keywords)

        if has_spatial_keyword or has_function_keyword:
            reward += w_spatial_function

        # ---- 2. Reasoning keyword reward ----
        reasoning_count = sum(k in r for k in reasoning_keywords)
        reasoning_reward = min(reasoning_count * w_reasoning, max_reasoning_reward)
        reward += reasoning_reward

        # ---- 3. Reward if [SEG] appears AFTER any spatial keyword ----
        # We check the index of first spatial keyword vs index of [SEG]
        seg_index = r.find("[seg]")

        if seg_index != -1:
            # find earliest occurrence of any spatial keyword
            spatial_positions = [r.find(k) for k in spatial_keywords if r.find(k) != -1]
            if len(spatial_positions) > 0:
                first_spatial_index = min(spatial_positions)

                # [SEG] must appear later in the text
                if seg_index > first_spatial_index:
                    reward += w_seg_after_spatial

        rewards.append(reward)

    return rewards
    


def combined_reward(completions, answers, pred_masks, gt_masks, weight=(1,1,10,0), **kwargs):
    format_score= format_reward_cal(completions=completions)
    correctness_score= correctness_reward_cal(completions=completions, answers=answers)
    iou_score= iou_reward_cal(predictions=pred_masks, ground_truths=gt_masks)
    key_word_score = key_words_reward_cal(completions=completions, answers=answers)
    combined_rewards=[]
    for format_reward, correctness_reward, iou_reward, key_word_reward in zip(format_score, correctness_score, iou_score, key_word_score):
        combined_reward= weight[0]*format_reward + weight[1]*correctness_reward + weight[2]*iou_reward + weight[3]*key_word_reward
        combined_rewards.append(combined_reward)
    return combined_rewards