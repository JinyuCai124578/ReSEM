# read a .pth
# import torch
# import os
# # file_path="/home/bingxing2/ailab/caijinyu/LISA/runs/lisa-em-bio-ft-500step-lr5e-6/meta_log_giou0.276_ciou0.515.pth"
# folder_path="/home/bingxing2/ailab/caijinyu/LISA/runs/lisa-em-all-ft-500step-lr1e-6"
# for filename in os.listdir(folder_path):
#     if filename.endswith(".pth"):
#         with open(os.path.join(folder_path, filename), 'rb') as f:
#             data = torch.load(f)
#             print(filename,data['epoch'])


import json

import json

# 输入文件路径
sft_result_path = '/home/caijinyu/LISA/vis_output/lisa-em-qa-lora-lr1e-5/all_eval_result.json'
grpo_result_path = '/home/caijinyu/LISA/vis_output/lisa-em-qa-lora-lr1e-5-grpo-1e-6-12kwreward/all_eval_result.json'
output_path = '/home/caijinyu/LISA/chat_sample_better.json'

# 读取JSON文件
with open(sft_result_path, 'r') as f:
    sft_data = json.load(f)
    
with open(grpo_result_path, 'r') as f:
    grpo_data = json.load(f)

# 确保两个文件长度相同
assert len(sft_data) == len(grpo_data), "两个JSON文件长度不一致"

# 筛选grpo的f1值更高的例子并保存比较结果
better_samples = []
for sft_item, grpo_item in zip(sft_data, grpo_data):
    if grpo_item['f1'] > sft_item['f1']:
        # 创建一个包含双方结果的字典
        comparison_item = {
            'image': grpo_item['image'],  # 保持原始图像路径
            'prompt': grpo_item['prompt'],  # 保持原始prompt
            'class': grpo_item['class'],  # 保持原始类别
            'sft_answer': sft_item['answer'],
            'grpo_answer': grpo_item['answer'],
            'sft_f1': sft_item['f1'],
            'grpo_f1': grpo_item['f1'],
            'sft_iou': sft_item['iou'],
            'grpo_iou': grpo_item['iou']
        }
        better_samples.append(comparison_item)

# 保存结果
with open(output_path, 'w') as f:
    json.dump(better_samples, f, indent=2)

print(f"筛选完成，共找到{len(better_samples)}个grpo表现更好的样本")
print(f"结果已保存到 {output_path}，包含双方的answer、f1和iou比较")