# lisa with qwen2.5vl
# copied from https://anonymous.4open.science/r/PathChat-Seg-3116/model/QWSA.py


import os
import numpy as np
from typing import List, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from model.segment_anything import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
from model.segment_anything.utils.transforms import ResizeLongestSide

# --- Helper functions (dice_loss, sigmoid_ce_loss, preprocess_image_for_sam) remain unchanged ---
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
    inputs = inputs.sigmoid().flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    return loss.sum() / (num_masks + 1e-8)

def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)

def preprocess_image_for_sam(image: torch.Tensor, image_size: int = 1024) -> torch.Tensor:
    """
    Preprocess image for SAM model
    Args:
        image: Input image tensor, possible shapes:
               - [B, C, H, W]: Batch of images
               - [C, H, W]: Single image
               - [B, 1, C, H, W]: Qwen format image
        image_size: Expected image size for SAM
    Returns:
        Processed image tensor [B, C, H, W]
    """
    # Print debug information
    # print(f"Original image shape: {image.shape}")
    
    # Handle different input shapes
    if image.dim() == 5:  # [B, 1, C, H, W] -> [B, C, H, W]
        image = image.squeeze(1)
    elif image.dim() == 3:  # [C, H, W] -> [1, C, H, W]
        image = image.unsqueeze(0)
    elif image.dim() == 1:
        raise ValueError(f"Invalid input image dimension: {image.shape}. Expected at least 3D [C, H, W]")
    
    # Ensure image is 4D [B, C, H, W]
    if image.dim() != 4:
        raise ValueError(f"Invalid processed image dimension: {image.shape}. Expected 4D [B, C, H, W]")
    
    # print(f"Processed image shape: {image.shape}")
    
    # SAM normalization parameters
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=image.device).view(1, -1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=image.device).view(1, -1, 1, 1)
    
    # If image is in [0,1] range, convert to [0,255]
    if image.max() <= 1.0:
        image = image * 255.0
    
    # Import ResizeLongestSide (ensure it's imported)
    from segment_anything.utils.transforms import ResizeLongestSide
    transform = ResizeLongestSide(image_size)
    
    image_sam_list = []
    for i in range(image.shape[0]):
        # Get single image [C, H, W]
        single_image = image[i]
        
        # Convert to numpy format [H, W, C]
        img_np = single_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Use SAM's resize transform
        img_resized = transform.apply_image(img_np)
        
        # Convert back to tensor format [C, H, W]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(image.device)
        image_sam_list.append(img_tensor)
    
    # Stack into batch [B, C, H, W]
    image_sam = torch.stack(image_sam_list, dim=0).float()
    
    # SAM normalization
    image_sam = (image_sam - pixel_mean) / pixel_std
    
    # Pad to specified size
    _, _, h, w = image_sam.shape
    padh = image_size - h
    padw = image_size - w
    image_sam = F.pad(image_sam, (0, padw, 0, padh))
    
    # print(f"Final SAM image shape: {image_sam.shape}")
    return image_sam


class QWSAForCausalLM(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        # 保存必要参数，不创建模型
        self.seg_token_idx = kwargs.get("seg_token_idx")
        if self.seg_token_idx is None:
            raise ValueError("seg_token_idx must be provided.")

        self.image_size = kwargs.get("image_size", 1024)
        self.ce_loss_weight = kwargs.get("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", 0.5)
        self.bce_loss_weight = kwargs.get("bce_loss_weight", 2.0)
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.out_dim = kwargs.get("out_dim", 256)
        self.train_mask_decoder = kwargs.get("train_mask_decoder", True)

        # 不初始化 visual_model 和 text_hidden_fcs
        # self.visual_model = None
        # self.text_hidden_fcs = None
        # self.initialize_lisa_modules()


    def initialize_lisa_modules(self):
        """在 from_pretrained 调用之后执行，避免 warning"""

        # ======================
        # 1. 初始化 SAM 模型
        # ======================
        if self.vision_pretrained and 'vit_h' in self.vision_pretrained:
            self.visual_model = build_sam_vit_h(None)
        elif self.vision_pretrained and 'vit_l' in self.vision_pretrained:
            self.visual_model = build_sam_vit_l(None)
        else:
            self.visual_model = build_sam_vit_h(None)

        self.visual_model.to_empty(device="cpu")

        # 加载权重
        sam_state = torch.load(self.vision_pretrained, map_location="cpu")
        self.visual_model.load_state_dict(sam_state, strict=True)

        assert not any(torch.isnan(p).any() for _, p in self.visual_model.image_encoder.named_parameters()), \
            "NaN weights found"

        # freeze encoder
        for p in self.visual_model.image_encoder.parameters():
            p.requires_grad = False

        # train mask decoder if needed
        if self.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for p in self.visual_model.mask_decoder.parameters():
                p.requires_grad = True


        # ======================
        # 2. 初始化 text_hidden_fcs
        # ======================
        in_dim = self.config.hidden_size
        out_dim = self.out_dim

        self.text_hidden_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
            )
        ])

        for p in self.text_hidden_fcs.parameters():
            p.requires_grad = True

        self.text_hidden_fcs.train()

        print("LISA modules initialized successfully.")

    @torch.no_grad()
    def get_sam_image_embeddings(self, images_for_sam: torch.Tensor):
        """
        Use SAM image encoder to get image embeddings.
        """
        # sam_input_images = preprocess_image_for_sam(images_for_sam, self.image_size).to(self.dtype) # preprocessed in dataset ReasonSegDatasetQWSA_EM
        sam_input_images=images_for_sam.to(self.dtype)
        return self.visual_model.image_encoder(sam_input_images)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs or 'super' in kwargs:
            # kwargs 仅传入 pixel_values,input_ids,labels,attention_mask
            # key_list=list(kwargs.keys())
            # for key in key_list:
            #     if key not in ['pixel_values', 'input_ids', 'labels', 'attention_mask']:
            #         kwargs.pop(key, None)
            kwargs.pop('super', None)
            return super().forward(**kwargs) # self.generate -> super().forward
        if 'grpo' in kwargs and kwargs['grpo']:
            return self.model_forward_grpo(**kwargs)
        return self.model_forward(**kwargs)
    
    def model_forward(self, **kwargs):
        """
        Forward propagation logic.
        Handle text generation and mask prediction.
        """
        # If in inference mode, directly call parent method without any modifications
        # if not self.training:
        #     return super().forward(**kwargs)

        # --- Training mode logic below ---
        
        # Separate custom parameters from kwargs
        images_for_sam = kwargs.pop('images')
        masks_list = kwargs.pop('masks_list')
        label_list = kwargs.pop('label_list')
        resize_list = kwargs.pop('resize_list')
        inference= kwargs.pop('inference', False)
        kwargs.pop('image_paths', None)
        kwargs.pop('questions_list', [])
        kwargs.pop('sampled_classes_list', [])
        kwargs.pop('offset', None)
        kwargs.pop('classes_list', None)
        kwargs.pop('prompt_ids', None)
        kwargs.pop('attention_masks_prompts', None)

        # 1. Get language model loss and hidden states
        kwargs['output_hidden_states'] = True
        # pixel_values,input_ids,labels,attention_mask
        outputs = super().forward(**kwargs)
        ce_loss = outputs.loss
        
        # 2. Get mask for [SEG] tokens
        hidden_states = outputs.hidden_states[-1]
        input_ids = kwargs["input_ids"]
        seg_token_mask = (input_ids == self.seg_token_idx)

        # If mask size needs adjustment
        if hidden_states.shape[1] > seg_token_mask.shape[1]:
            seg_token_mask = F.pad(seg_token_mask, (0, hidden_states.shape[1] - seg_token_mask.shape[1]))

        # Extract and project [SEG] embeddings
        pred_text_embeddings = self.text_hidden_fcs[0](hidden_states[seg_token_mask])

        # 3. Get SAM image embeddings
        sam_image_embeds = self.get_sam_image_embeddings(images_for_sam)

        # Use projected text embeddings and SAM image embeddings to predict masks
        pred_masks = []
        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_counts.device).long(), seg_token_counts.cumsum(-1)])

        # Process each segmentation token to generate masks
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            if start_i >= end_i: continue

            current_text_embeds = pred_text_embeddings[start_i:end_i]
            
            # Use prompt_encoder to get sparse and dense embeddings
            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=current_text_embeds.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(current_text_embeds.dtype)

            # Use mask_decoder to get segmentation masks
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=sam_image_embeds[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        # Calculate mask loss
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_valid_masks = 0
        # import pdb;pdb.set_trace()
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }
        
        for i, pred_mask_item in enumerate(pred_masks):
            gt_mask = masks_list[i]
            if gt_mask.numel() == 0 or pred_mask_item.numel() == 0:
                continue
            
            # Ensure predicted and GT mask counts match
            if pred_mask_item.shape[0] != gt_mask.shape[0]:
                logging.warning(f"Batch {i} mask mismatch: pred {pred_mask_item.shape[0]}, gt {gt_mask.shape[0]}")
                continue

            num_valid_masks += pred_mask_item.shape[0]
            mask_bce_loss += sigmoid_ce_loss(pred_mask_item, gt_mask, num_masks=1) * pred_mask_item.shape[0]
            mask_dice_loss += dice_loss(pred_mask_item, gt_mask, num_masks=1) * pred_mask_item.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_valid_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_valid_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Total loss
        total_loss = self.ce_loss_weight * ce_loss + mask_loss
        
        return {"loss": total_loss, 
                "ce_loss": ce_loss, 
                "mask_loss": mask_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss}
    
    def model_forward_grpo(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_masks: torch.LongTensor,
        pad_token_id: int ,
        eos_token_id: int,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        do_sample: bool=True,
        early_stopping: bool = False,
        **kwargs,
    ):
        """
        Forward propagation logic.
        Handle text generation and mask prediction.
        """
        # If in inference mode, directly call parent method without any modifications
        # if not self.training:
        #     return super().forward(**kwargs)

        # --- Training mode logic below ---
        
        # Separate custom parameters from kwargs
        outputs=self.generate(
            # images=images,
            inputs=input_ids,
            attention_mask=attention_masks,
            max_new_tokens=max_new_tokens,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            early_stopping=early_stopping)
        
        # 2. Get mask for [SEG] tokens
        # hidden_states = outputs.hidden_states[-1]
        if type(outputs.hidden_states[-1]) is tuple:
            if outputs.hidden_states[-1][-1].shape[1] ==1:
                hidden_states=torch.cat(outputs.hidden_states[-1], dim=1)
            else:
                hidden_states = outputs.hidden_states[-1][-1]
        elif outputs.hidden_states[-1].shape[1] ==1:
            hidden_states=torch.cat(outputs.hidden_states, dim=1)
        else:
            hidden_states = outputs.hidden_states[-1]
        # input_ids = kwargs["input_ids"]
        output_ids = outputs.sequences
        seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
        for i in range(seg_token_mask.shape[0]):
            if seg_token_mask[i].sum() == 0:
                # 把 seg_token_mask[i] 的最后一个 token 设为 True
                seg_token_mask[i][-1] = True

        # If mask size needs adjustment
        if hidden_states.shape[1] > seg_token_mask.shape[1]:
            seg_token_mask = F.pad(seg_token_mask, (0, hidden_states.shape[1] - seg_token_mask.shape[1]))
        elif hidden_states.shape[1] < seg_token_mask.shape[1]:
            seg_token_mask= seg_token_mask[:, -hidden_states.shape[1]:]

        # Extract and project [SEG] embeddings
        pred_text_embeddings = self.text_hidden_fcs[0](hidden_states[seg_token_mask])

        # 3. Get SAM image embeddings
        sam_image_embeds = self.get_sam_image_embeddings(images)

        # Use projected text embeddings and SAM image embeddings to predict masks
        pred_low_res_masks = []
        seg_token_counts = seg_token_mask.int().sum(-1)
        seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_counts.device).long(), seg_token_counts.cumsum(-1)])

        # Process each segmentation token to generate masks
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            if start_i >= end_i: continue

            current_text_embeds = pred_text_embeddings[start_i:end_i]
            
            # Use prompt_encoder to get sparse and dense embeddings
            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=current_text_embeds.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(current_text_embeds.dtype)

            # Use mask_decoder to get segmentation masks
            low_res_masks, _ = self.visual_model.mask_decoder(
                image_embeddings=sam_image_embeds[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_low_res_masks.append(low_res_masks[0])
        pred_low_res_masks=torch.stack(pred_low_res_masks, dim=0)


        B, L = output_ids.shape
        if L < max_new_tokens + input_ids.shape[1]:
            pad_len = (max_new_tokens + input_ids.shape[1]) - L
            pad = torch.full(
                (B, pad_len),
                pad_token_id,
                dtype=output_ids.dtype,
                device=output_ids.device,
            )
            output_ids = torch.cat([output_ids, pad], dim=1)
        return {
            'output_ids':output_ids, 
            'pred_low_res_masks':pred_low_res_masks, 
        }


def evaluate(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        resize_list,
        original_size_list,
        max_new_tokens: int = None,
    ):
        """
        Forward propagation logic.
        Handle text generation and mask prediction.
        """
        # If in inference mode, directly call parent method without any modifications
        # if not self.training:
        #     return super().forward(**kwargs)

        # --- Training mode logic below ---
        
        # Separate custom parameters from kwargs
        with torch.no_grad():
            outputs=self.generate(
                # images=images,
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,)
            
            # 2. Get mask for [SEG] tokens
            # hidden_states = outputs.hidden_states[-1]
            if type(outputs.hidden_states[-1]) is tuple:
                if outputs.hidden_states[-1][-1].shape[1] ==1:
                    hidden_states=torch.cat(outputs.hidden_states[-1], dim=1)
                else:
                    hidden_states = outputs.hidden_states[-1][-1]
            elif outputs.hidden_states[-1].shape[1] ==1:
                hidden_states=torch.cat(outputs.hidden_states, dim=1)
            else:
                hidden_states = outputs.hidden_states[-1]
            # input_ids = kwargs["input_ids"]
            output_ids = outputs.sequences
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            for i in range(seg_token_mask.shape[0]):
                if seg_token_mask[i].sum() == 0:
                    # 把 seg_token_mask[i] 的最后一个 token 设为 True
                    seg_token_mask[i][-1] = True

            # If mask size needs adjustment
            if hidden_states.shape[1] > seg_token_mask.shape[1]:
                seg_token_mask = F.pad(seg_token_mask, (0, hidden_states.shape[1] - seg_token_mask.shape[1]))
            elif hidden_states.shape[1] < seg_token_mask.shape[1]:
                seg_token_mask= seg_token_mask[:, -hidden_states.shape[1]:]

            # Extract and project [SEG] embeddings
            pred_text_embeddings = self.text_hidden_fcs[0](hidden_states[seg_token_mask])

            # 3. Get SAM image embeddings
            sam_image_embeds = self.get_sam_image_embeddings(images)

            # Use projected text embeddings and SAM image embeddings to predict masks
            pred_low_res_masks = []
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_counts.device).long(), seg_token_counts.cumsum(-1)])

            # Process each segmentation token to generate masks
            pred_masks = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                if start_i >= end_i: continue

                current_text_embeds = pred_text_embeddings[start_i:end_i]
                
                # Use prompt_encoder to get sparse and dense embeddings
                sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None, text_embeds=current_text_embeds.unsqueeze(1)
                )
                sparse_embeddings = sparse_embeddings.to(current_text_embeds.dtype)

                # Use mask_decoder to get segmentation masks
                low_res_masks, _ = self.visual_model.mask_decoder(
                    image_embeddings=sam_image_embeds[i].unsqueeze(0),
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = self.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])
            
            return output_ids, pred_masks