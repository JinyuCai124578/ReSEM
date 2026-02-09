#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#    from  https://github.com/LLaVA-VL/LLaVA-NeXT/blob/e9835311c6f515a13702eb7a7750fcd936f65ed8/llava/model/language_model/llava_qwen.py

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# 获取 llava 目录的绝对路径
import sys
import os
llava_path = os.path.abspath('ReSEM/model')
# 将 llava 目录添加到 sys.path
if llava_path not in sys.path:
    sys.path.insert(0, llava_path)
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = True,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            # llava version
            (
             input_ids, 
             attention_mask, 
             past_key_values, 
             inputs_embeds, 
             labels
             ) = self.prepare_inputs_labels_for_multimodal(
                 input_ids, attention_mask, past_key_values, labels, images
            )
            # llava1.5 version
            # (input_ids, 
            #  position_ids, 
            #  attention_mask, 
            #  past_key_values, 
            #  inputs_embeds, 
            #  labels) = self.prepare_inputs_labels_for_multimodal_1_5(
            #      input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes=image_sizes
            #      )
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            # refer to llava_llama.py LlavaLlamaForCausalLM forward function 
            loss=None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous() # labels.shape: [3, 113], logits.shape: [1, 831, 151651]
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            if self.training:
                output_hidden_states = outputs.hidden_states
            else:
                output_hidden_states = hidden_states
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=output_hidden_states,  # outputs.hidden_states,
                attentions=outputs.attentions,
            )

        else:
            return super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None, # # You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     images: Optional[torch.Tensor] = None,
    #     image_sizes: Optional[torch.Tensor] = None,
    #     modalities: Optional[List[str]] = ["image"],
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     position_ids = kwargs.pop("position_ids", None)
    #     attention_mask = kwargs.pop("attention_mask", None)
    #     if "inputs_embeds" in kwargs:
    #         raise NotImplementedError("`inputs_embeds` is not supported")

    #     if images is not None:
    #         import pdb; pdb.set_trace()
    #         (
    #          inputs, 
    #          attention_mask, 
    #          past_key_values, 
    #          inputs_embeds, 
    #          labels
    #          ) = self.prepare_inputs_labels_for_multimodal(
    #              inputs, attention_mask, None, None, images
    #         )
    #         # (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
    #     else:
    #         inputs_embeds = self.get_model().embed_tokens(inputs)

    #     return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, images=None, **kwargs):
        # images = kwargs.pop("images", None)
        # image_sizes = kwargs.pop("image_sizes", None)
        # inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        # if images is not None:
        #     inputs["images"] = images
        # if image_sizes is not None:
        #     inputs["image_sizes"] = image_sizes
        # return inputs
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)