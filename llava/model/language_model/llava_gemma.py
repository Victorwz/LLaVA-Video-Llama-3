#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Gemma2Config, Gemma2Model, Gemma2ForCausalLM, Gemma2ForSequenceClassification

from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """        
        # def sim_matrix(a, b, eps=1e-6):
        #     """
        #     added eps for numerical stability
        #     """
        #     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        #     a_norm = a / torch.clamp(a_n, min=eps)
        #     b_norm = b / torch.clamp(b_n, min=eps)
        #     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        #     return sim_mt

        # cosine_sim = torch.mm(projections, projections.T)
        # dis = cosine_sim[~torch.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
        # dis = dis / self.temperature
        # cosine_sim = cosine_sim / self.temperature
        # # apply exp to elements
        # dis = torch.exp(dis)
        # cosine_sim = torch.exp(cosine_sim)
        # row_sum = []
        # for i in range(projections.shape[0]):
        #     row_sum.append(sum(dis[i]))
        # # calculate outer sum
        # contrastive_loss = 0
        # for i in range(projections.shape[0]):
        #     n_i = targets.tolist().count(targets[i]) - 1
        #     inner_sum = 0
        #     # calculate inner sum
        #     for j in range(projections.shape[0]):
        #         if targets[i] == targets[j] and i != j:
        #             inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
        #     if n_i != 0:
        #         contrastive_loss += (inner_sum / (-n_i))
        #     else:
        #         contrastive_loss += 0
        # return contrastive_loss

        # device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        # projections = torch.nn.functional.normalize(projections, p=2, dim=1)
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # import pdb
        # pdb.set_trace()
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(projections.device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(projections.device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

class LlavaGemmaConfig(Gemma2Config):
    model_type = "llava_gemma"
    # scl_temperature = 0.7
    # loss_weight_scalar = 0.2

class LlavaGemmaModel(LlavaMetaModel, Gemma2Model):
    config_class = LlavaGemmaConfig

    def __init__(self, config: Gemma2Config):
        super(LlavaGemmaModel, self).__init__(config)


class LlavaGemmaForCausalLM(Gemma2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaGemmaConfig

    def __init__(self, config):
        super(Gemma2ForCausalLM, self).__init__(config)
        self.model = LlavaGemmaModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
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
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

class LlavaGemmaClassifier(Gemma2ForSequenceClassification, LlavaMetaForCausalLM):
    config_class = LlavaGemmaConfig

    def __init__(self, config):
        super(Gemma2ForSequenceClassification, self).__init__(config)
        self.model = LlavaGemmaModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.num_labels = config.num_labels
        # self.scl_temperature = config.scl_temperature
        
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        # self.embd_ln = nn.LayerNorm(config.hidden_size)

        # the loss function is weighted sum between CE loss and SCL loss
        # self.loss_weight_scalar = config.loss_weight_scalar

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(input_ids.device)
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1

        if False:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
            )
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)

            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            pooled_logits = sum_embeddings / sum_mask
        else:
            pooled_logits = hidden_states[torch.arange(batch_size, device=sequence_lengths.device), sequence_lengths]
            # normed_logits = self.embd_ln(pooled_logits)

        classification_logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            
            if self.num_labels == 1:
                labels = labels.type(classification_logits.dtype).to(classification_logits.device)
                mse_loss_fct = nn.MSELoss()
                loss = mse_loss_fct(classification_logits.squeeze(), labels.squeeze())
            else:
                labels = labels.type(torch.int64).to(classification_logits.device)
                classification_loss_fct = nn.CrossEntropyLoss()
                loss = classification_loss_fct(classification_logits.view(-1, self.num_labels), labels.view(-1))
            
            # calculating contrastive loss for contrastive learning
            # scl_loss_fct = SupervisedContrastiveLoss(self.scl_temperature)
            # scl_loss = scl_loss_fct(classification_logits, labels.view(-1))

            # loss = loss #(self.loss_weight_scalar * scl_loss) + (1 - self.loss_weight_scalar) * loss
        
        if not return_dict:
            output = (classification_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=classification_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )        

        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            # deal with multi images

            concat_images = torch.stack(images) # num_images, 3, 336, 336
            image_features = self.encode_images(concat_images)
            # if type(images) is list:
            #     images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # concat_images = torch.cat([image for image in images], dim=0)
            # image_features = self.encode_images(concat_images)
            # split_sizes = [image.shape[0] for image in images]
            # image_features = torch.split(image_features, split_sizes, dim=0)
            # mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            # image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            # if mm_patch_merge_type == 'flat':
            #     image_features = [x.flatten(0, 1) for x in image_features]
            # elif mm_patch_merge_type.startswith('spatial'):
            #     new_image_features = []
            #     for image_idx, image_feature in enumerate(image_features):
            #         if image_feature.shape[0] > 1:
            #             base_image_feature = image_feature[0]
            #             image_feature = image_feature[1:]
            #             height = width = self.get_vision_tower().num_patches_per_side
            #             assert height * width == base_image_feature.shape[0]
            #             if image_aspect_ratio == 'anyres':
            #                 num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
            #                 image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            #             else:
            #                 raise NotImplementedError
            #             if 'unpad' in mm_patch_merge_type:
            #                 image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            #                 image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            #                 image_feature = unpad_image(image_feature, image_sizes[image_idx])
            #                 image_feature = torch.cat((
            #                     image_feature,
            #                     self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
            #                 ), dim=-1)
            #                 image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            #             else:
            #                 image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
            #                 image_feature = image_feature.flatten(0, 3)
            #             image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            #         else:
            #             image_feature = image_feature[0]
            #             if 'unpad' in mm_patch_merge_type:
            #                 image_feature = torch.cat((
            #                     image_feature,
            #                     self.model.image_newline[None].to(image_feature.device)
            #                 ), dim=0)
            #         new_image_features.append(image_feature)
            #     image_features = new_image_features
            # else:
            #     raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # import time
            # start = time.time()
            image_features = self.encode_images(images)

            # print("encoding time is {:.4f}".format(time.time() - start))
            # print(image_features.shape)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        
        new_input_embeds = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            split_sizes = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes.append(image_token_indices[i+1] - image_token_indices[i] - 1)
            
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, labels


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_gemma", LlavaGemmaConfig)
AutoModelForCausalLM.register(LlavaGemmaConfig, LlavaGemmaForCausalLM)
