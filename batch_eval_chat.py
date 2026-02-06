import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.LISA_qwen import LISAQwenForCausalLM
from model.llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, EXPLANATORY_QUESTION_LIST)
import pdb
import random
from train_ds import validate, validate_text
from tqdm import tqdm
import gc
import json
import deepspeed
from PIL import Image

import traceback

random.seed(42)

def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    print()
    pdb.pm()

sys.excepthook = info

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/model/cache/models--xinlai--LISA-13B-llama2-v1/snapshots/b89000be11ad0a45512745a15063f2f6af1d9a5c")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="/mnt/shared-storage-user/caijinyu/model/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=False)
    parser.add_argument("--chat_json", default="/home/caijinyu/LISA/chat_sample.json", type=str, required=False)
    parser.add_argument("--dataset_dir", default='/mnt/shared-storage-user/caijinyu/data', type=str)
    parser.add_argument("--reason_seg_data", default="organelle||plantorgan||cremi||material", type=str)
    parser.add_argument("--use_gpt_qa", action="store_true", default=True)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)

    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    # 清理cuda cache
    torch.cuda.empty_cache()
    args = parse_args(args)
    args.vis_save_path = os.path.join(args.vis_save_path, args.version.split("/")[-1])
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir="/home/bingxing2/ailab/group/ai4neuro/EM_segmentation/model/lisa",
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

    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    if "qwen" in args.version:
        model = LISAQwenForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
        )
    else:
        model = LISAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
        )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    if 'siglip' in model.config.vision_tower:
        clip_image_processor = SigLipVisionTower(model.config.vision_tower).image_processor
    else:
        clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        
    transform = ResizeLongestSide(args.image_size)

    # if args.weight != "" :
    #     if "lora" in args.weight.lower():
    #         state_dict = torch.load(args.weight, map_location="cpu")
    #         model_dict = model.state_dict()
    #         state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    #         model.load_state_dict(state_dict, strict=False)
    #         print("Loaded LORA weights")
    #     else:
    #         state_dict = torch.load(args.weight, map_location="cpu")
    #         model.load_state_dict(torch.load(args.weight))
    if args.weight != "":
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

    from utils.dataset import collate_fn_grpo, ValDataset_EM
    from functools import partial
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]
    val_dataset = ValDataset_EM(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.reason_seg_data+"_val",
                args.image_size,
                use_gpt_qa=args.use_gpt_qa,
            )
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(
                collate_fn_grpo,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    
    model.eval()
    # giou, ciou, text_metric = validate_text(val_loader, model, 0, None,tokenizer, args)
    # print("GIoU: ", giou, ";CIoU: ", ciou, ";Text_metric: ", text_metric)
    # metric_save_path=os.path.join(args.vis_save_path,'metric.json')
    # with open(metric_save_path, 'w') as f:
    #     json.dump({"GIoU": giou, "CIoU": ciou, "Text_metric": text_metric}, f, indent=4)

    def chat(prompt,image_path,class_name,mask_path,color_id):
        if "ceramic" in mask_path.lower() or "nanoparticle" in mask_path.lower():
            gt_mask=np.array(Image.open(mask_path).convert('L'))==int(color_id) if color_id is not None \
                    else np.array(Image.open(mask_path).convert('L'))!=0
        else:
            gt_mask=np.array(Image.open(mask_path))==int(color_id) if color_id is not None \
                    else np.array(Image.open(mask_path))!=0
    
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt + random.choice(EXPLANATORY_QUESTION_LIST)
        # prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt # + random.choice(EXPLANATORY_QUESTION_LIST)
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            return
        
        if "tiff" in image_path:
            image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
            image = (image-np.min(image))/(np.max(image)-np.min(image)) *255
            image_np=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        else:
            image_np = cv2.imread(image_path)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        # if input_ids.shape[1]==1:
        # import pdb; pdb.set_trace()
        output_ids, pred_masks = model.evaluate(
            image_clip, # torch.Size([1, 3, 224, 224])
            image, # torch.Size([1, 3, 1024, 1024])
            input_ids, # torch.Size([1, 76])
            resize_list, #[(689, 1024)]
            original_size_list, # [(3230, 4800)]
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        f1=1.0 if class_name.replace("_", " ").lower() in text_output.lower() else 0.0

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            intersection=np.sum(pred_mask * gt_mask)
            union=np.sum(pred_mask) + np.sum(gt_mask)
            iou=intersection / (union+1e-5)

            # save_path = "{}/{}_mask_{}.jpg".format(
            #     args.vis_save_path, class_name, i
            # )
            # cv2.imwrite(save_path, pred_mask * 100)
            # print("{} has been saved.".format(save_path))

            # save_path = "{}/{}_masked_img_{}.jpg".format(
            #     args.vis_save_path, class_name, i
            # )
            # save_img = image_np.copy()
            # save_img[pred_mask] = (
            #     image_np * 0.5
            #     + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            # )[pred_mask]
            # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(save_path, save_img)
            # print("{} has been saved.".format(save_path))
        return {"image": image_path, "prompt": prompt, "class": class_name, "answer": text_output.split('ASSISTANT: ')[-1], "f1": f1, "iou": iou}
    
    # sample_dict = json.load(open(args.chat_json))
    # result_json=[]
    # for i in range(len(sample_dict)):
    #     result=chat(sample_dict[i]["prompt"],sample_dict[i]["image"],sample_dict[i]["class"])
    #     result_json.append(result)
    
    # result_save_path = os.path.join(args.vis_save_path,args.chat_json.split("/")[-1])
    # with open(result_save_path,"w") as f:
    #     json.dump(result_json,f,indent=4)
    val_data_list = val_dataset.json_data_list
    result_json=[]
    for item in tqdm(val_data_list):
        image_path=item['image_path']
        for shape in item['shapes']:
            mask_path=image_path.replace(shape['image_name'],shape['mask_name'])
            color_id=shape.get('color_id',None)
            class_name=shape['class_name']
            for qa in shape['qa_list']:
                prompt=qa['question']
                result=chat(prompt=prompt,
                            image_path=image_path,
                            class_name=class_name,
                            mask_path=mask_path,
                            color_id=color_id)
                result_json.append(result)
    result_save_path = os.path.join(args.vis_save_path,'all_eval_result.json')
    with open(result_save_path,"w") as f:
        json.dump(result_json,f,indent=4)


if __name__ == "__main__":
    main(sys.argv[1:])
