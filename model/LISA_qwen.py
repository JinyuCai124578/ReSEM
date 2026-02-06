from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_qwen import (LlavaQwenForCausalLM,
                                                   LlavaQwenModel)
from .segment_anything import build_sam_vit_h, build_sam_vit_l
import pdb


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaQwenMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaQwenMetaModel, self).__init__(config) # why is this necessary?

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        if self.vision_pretrained and 'vit_h' in self.vision_pretrained:
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
        elif self.vision_pretrained and 'vit_l' in self.vision_pretrained:
            self.visual_model = build_sam_vit_l(self.vision_pretrained)
        else:
            self.visual_model = build_sam_vit_h(self.vision_pretrained)

        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

        # ! yyc add to load fcs weight
        # if hasattr(self.config, "train_mask_decoder"):
        #     fcs_weight = torch.load("./Lisa_tuned_fcs.bin")
        #     self.text_hidden_fcs.load_state_dict(fcs_weight, strict=True, assign=True)

        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaQwenModel(LisaQwenMetaModel, LlavaQwenModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaQwenModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAQwenForCausalLM(LlavaQwenForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaQwenModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs or 'super' in kwargs:
            return super().forward(**kwargs) # self.generate -> super().forward
        if 'grpo' in kwargs and kwargs['grpo']:
            return self.model_forward_grpo(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # 提取图像特征  [batch_size, embedding_dim]
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        
        # 处理文本输入 seg_token_mask 用于标记哪些 token 是分割相关的特殊 token
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            #  调用父类的 forward 方法，生成文本的隐藏状态。最终将隐藏状态存储在 output_hidden_states 中
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            # train时批量处理
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        # 用hidden_states中的最后一层输入到SAM中的embeddings
        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front) 不止255
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], last_hidden_state.shape[1]-seg_token_mask.shape[1])).bool().cuda(), seg_token_mask],
            dim=1,
        )
        pred_embeddings = last_hidden_state[seg_token_mask] # [1, 839, 256], [2, 368]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        # pdb.set_trace()  # 设置断点
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list
        # pdb.set_trace()  # 设置断点
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = ce_loss-ce_loss
        mask_dice_loss = ce_loss-ce_loss
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            # assert (
            #     gt_mask.shape[0] == pred_mask.shape[0]
            # ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
            #     gt_mask.shape, pred_mask.shape
            # )
            if gt_mask.shape[0] != pred_mask.shape[0]:
                continue  # 如果形状不匹配，则跳过当前循环迭代 没辙了

            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    
    def model_forward_grpo(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
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
        outputs = self.generate(
            images=images_clip,
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_new_tokens=max_new_tokens,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            early_stopping=early_stopping
        )
        # import pdb; pdb.set_trace()
        output_hidden_states = outputs.hidden_states[-1][-1] if type(outputs.hidden_states[-1]) is tuple else outputs.hidden_states[-1]
        if type(outputs.hidden_states[-1]) is tuple:
            output_hidden_states = outputs.hidden_states[-1][-1]
        elif outputs.hidden_states[-1].shape[1] ==1:
            output_hidden_states=torch.cat(outputs.hidden_states, dim=1)
        else:
            output_hidden_states = outputs.hidden_states[-1]
    

        output_ids = outputs.sequences # torch.Size([4, 166])
        seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
        for i in range(seg_token_mask.shape[0]):
            if seg_token_mask[i].sum() == 0:
                # 把 seg_token_mask[i] 的最后一个 token 设为 True
                seg_token_mask[i][-1] = True
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [
                torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                seg_token_mask,
            ],
            dim=1,
        )

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states)) 

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # import pdb; pdb.set_trace()
        pred_embeddings = last_hidden_state[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        image_embeddings = self.get_visual_embs(images)

        multimask_output = False
        pred_low_res_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_low_res_masks.append(low_res_masks[0])
            # pred_mask = self.model.visual_model.postprocess_masks(
            #     low_res_masks,
            #     input_size=resize_list[i],
            #     original_size=label_list[i],
            # )
            # pred_masks.append(pred_mask[:, 0])

        # prompt_length = input_ids.shape[1]
        # completion_ids=output_ids[:, prompt_length:]
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
            'output_ids':output_ids, # 需要pad成相同长度
            'pred_low_res_masks':pred_low_res_masks, # 需要变成张量
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            if outputs.hidden_states[-1].shape[1] ==1:
                output_hidden_states=torch.cat(outputs.hidden_states, dim=1)
            else:
                output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            # import pdb; pdb.set_trace()
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], last_hidden_state.shape[1]-seg_token_mask.shape[1])).bool().cuda(), seg_token_mask],
                dim=1,
            )
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
