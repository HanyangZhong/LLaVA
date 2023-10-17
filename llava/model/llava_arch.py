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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .audio_encoder.builder import build_audio_tower

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, AUDIO_TOKEN_INDEX,DEFAULT_AUDIO_PATCH_TOKEN,DEFAULT_AU_START_TOKEN,DEFAULT_AU_END_TOKEN,COMPACT_TOKEN_INDEX

# 新的元模型
class RobotGPTMetaModel:

    def __init__(self, config):
        super(RobotGPTMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
<<<<<<< Updated upstream
            self.mm_vision_projector = nn.Linear(config.mm_vision_hidden_size, config.vision_hidden_size)

        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=True)
            self.mm_audio_projector = nn.Linear(config.mm_audio_hidden_size, config.audio_hidden_size)
=======
            self.mm_vision_projector = nn.Linear(config.mm_vision_hidden_size, config.hidden_size)

        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=True)
            self.mm_audio_projector = nn.Linear(config.mm_audio_hidden_size, config.hidden_size)
>>>>>>> Stashed changes

    # 视觉头
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    # 音频头
    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    # 初始化视觉头
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        # 从pretrain_mm_mlp_adapter 变成 pretrain_mm_vision_mlp_adapter区分
        pretrain_mm_vision_mlp_adapter = model_args.pretrain_mm_vision_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        # 从use_mm_proj变成 use_mm_vision_proj
        self.config.use_mm_vision_proj = True
        # 从use_mm_proj变成 use_mm_vision_proj
<<<<<<< Updated upstream
        self.config.mm_vision_hidden_size = vision_tower.vision_hidden_size
=======
        self.config.mm_vision_hidden_size = vision_tower.hidden_size
>>>>>>> Stashed changes
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, 'mm_vision_projector'):
<<<<<<< Updated upstream
            self.mm_vision_projector = nn.Linear(self.config.mm_vision_hidden_size, self.config.vision_hidden_size)
=======
            self.mm_vision_projector = nn.Linear(self.config.mm_vision_hidden_size, self.config.hidden_size)
>>>>>>> Stashed changes

        if pretrain_mm_vision_mlp_adapter is not None:
            mm_vision_projector_weights = torch.load(pretrain_mm_vision_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_vision_projector.load_state_dict(get_w(mm_vision_projector_weights, 'mm_vision_projector'))

    # 初始化音频头
    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        # select layer 和feature暂时都在后面没用上，可以不管
        mm_audio_select_layer = model_args.mm_audio_select_layer
        mm_audio_select_feature = model_args.mm_audio_select_feature
        pretrain_mm_audio_mlp_adapter = model_args.pretrain_mm_audio_mlp_adapter

        self.config.mm_audio_tower = audio_tower
        # 音频头加载
        audio_tower = build_audio_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.audio_tower = [audio_tower]
        else:
            self.audio_tower = audio_tower

        # select layer 和 select feature 没用上，后面可以加
        self.config.use_mm_audio_proj = True
<<<<<<< Updated upstream
        self.config.audio_hidden_size = audio_tower.audio_hidden_size
=======
        self.config.mm_audio_hidden_size = audio_tower.hidden_size
>>>>>>> Stashed changes
        self.config.mm_audio_select_layer = mm_audio_select_layer
        self.config.mm_audio_select_feature = mm_audio_select_feature

        if not hasattr(self, 'mm_audio_projector'):
<<<<<<< Updated upstream
            self.mm_audio_projector_audio = nn.Linear(self.config.mm_audio_hidden_size, self.config.audio_hidden_size)
=======
            self.mm_audio_projector = nn.Linear(self.config.mm_audio_hidden_size, self.config.hidden_size)
>>>>>>> Stashed changes

        if pretrain_mm_audio_mlp_adapter is not None:
            mm_audio_projector_weights = torch.load(pretrain_mm_audio_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_audio_projector.load_state_dict(get_w(mm_audio_projector_weights, 'mm_audio_projector'))

# 加入多模态的元模型
class RobotGPTMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass
    
    # 视觉头
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    # 音频头
    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    # 编码图片
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_vision_projector(image_features)
        return image_features

    # 编码音频
    def encode_audio(self, audio):
        audio_features = self.get_model().get_audio_tower()(audio)
        audio_features = self.get_model().mm_audio_projector(audio_features)
        return audio_features

    # 图片多模态label加载
    def prepare_inputs_labels_for_visionmodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        # 视频头
        vision_tower = self.get_vision_tower()
        
        # 检查视觉头、图像输入或文本输入是否丢失    后加入音频头和音频是否丢失
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # 如果文本输入只有 1 个标记  视觉头和图片有  并且给出了过去的键值
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                # 扩展注意力掩码以包含过去的键值长度
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            
            # 返回输入 ID、注意掩码、过去的键值、无图像特征、无音频特征、标签
            return input_ids, attention_mask, past_key_values, None, labels

        # 将一个或一列表输入的图像转换成特征图
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # 遍历每个输入
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 查看是否包含特殊标记 TOKEN
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # 无图，无音频，只有文字 则将文字作为嵌入
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_vision_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                # 如果没有，则将文本输入附加到new_input_embeds
                # 将该id与label对其
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

           
            # 找到图片切片位置
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # 图片位置切片遍历
            while image_token_indices.numel() > 0:
                # 从image_features列表中获取对应的图像特征
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                # 在图像标记索引处拆分输入 ID 和标签
                if getattr(self.config, 'tune_mm_vision_mlp_adapter', False) and getattr(self.config, 'mm_vision_use_im_start_end', False):
                    # 可以选择在图像特征周围添加开始/结束标记
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    # 将图像特征附加到嵌入中
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))

                    if labels is not None:
                        # 在图像索引之前和之后附加文本嵌入
                        cur_new_labels.append(cur_labels[:image_token_start])
                        # 将图像区域标签设置为 IGNORE_INDEX
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # 嵌入任何剩余的文本标记
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        # 更新新的标签列表
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    # 直接在标记处取作为现在的嵌入
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                # 没有用 这些标记就手动累加
                if getattr(self.config, 'tune_mm_vision_mlp_adapter', False) and getattr(self.config, 'mm_vision_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_vision_mlp_adapter', False) and getattr(self.config, 'mm_vision_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    # 音频多模态label加载
    def prepare_inputs_labels_for_audiomodal(
        self, input_ids, attention_mask, past_key_values, labels, audio
    ):
        # 音频头
        audio_tower = self.get_audio_tower()
        
        # 检查视觉头、图像输入或文本输入是否丢失    后加入音频头和音频是否丢失
        if audio_tower is None or audio is None or input_ids.shape[1] == 1:
            # 如果文本输入只有 1 个标记  视觉头和图片有  并且给出了过去的键值
            if past_key_values is not None and audio_tower is not None and audio is not None and input_ids.shape[1] == 1:
                # 扩展注意力掩码以包含过去的键值长度
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)

            # 返回输入 ID、注意掩码、过去的键值、无图像特征、无音频特征、标签
            return input_ids, attention_mask, past_key_values, None, labels

        # 判断音频输入是列表还是一个
        # 检查audio维度是否5维，确定是否批量输入
        if type(audio) is list:
            # 将所有音频连接成一个张量
            # concat_audio = torch.cat([aud for aud in audio], dim=0)
            # # 对串联的音频进行编码
            # audio_features = self.encode_audio(concat_audio)
            # # 获取每个音频连接之前的尺寸
            # split_sizes = [aud.shape[0] for aud in audio]
            # # 将编码特征拆分回原批次
            # audio_features = torch.split(audio_features, split_sizes, dim=0)
            # # 最后将编码特征展平
            # audio_features = [x.flatten(0, 1) for x in audio_features]

            audio_features =[]
            for feature in audio:
                get_audio = feature['input_values']
                # print('input shape:',get_audio.shape)
                # print('feature shape:',feature)
                # print('mask shape:',feature['attention_mask'].shape)
                # print(**get_audio)
                output = self.encode_audio(get_audio)
                audio_features.append(output)

        else:
            audio_features = self.encode_audio(audio)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_audio_idx = 0
        print('making new inputs audio')
        # 遍历每个输入
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 查看是否包含特殊音频标记 TOKEN
            if (cur_input_ids == AUDIO_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # 无图，无音频，只有文字 则将文字作为嵌入
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_audio_projector(audio_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                # 如果没有，则将文本输入附加到new_input_embeds
                # 将该id与label对其
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_audio_idx += 1
                continue

            # 找到音频切片位置
            audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # 音频位置切片遍历
            while audio_token_indices.numel() > 0:
                # 从audio_features列表中获取对应的音频特征
                cur_audio_features = audio_features[cur_audio_idx]
                audio_token_start = audio_token_indices[0]
                # 在音频标记索引处拆分输入 ID 和标签
                if getattr(self.config, 'tune_mm_audio_mlp_adapter', False) and getattr(self.config, 'mm_audio_use_im_start_end', False):
                    # 可以选择在音频特征周围添加开始/结束标记
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:audio_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[audio_token_start-1:audio_token_start]))
                    # 将音频特征附加到嵌入中
                    cur_new_input_embeds.append(cur_audio_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[audio_token_start+1:audio_token_start+2]))

                    if labels is not None:
                        # 在音频索引之前和之后附加文本嵌入
                        cur_new_labels.append(cur_labels[:audio_token_start])
                        # 将音频区域标签设置为 IGNORE_INDEX
                        cur_new_labels.append(torch.full((cur_audio_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # 嵌入任何剩余的文本标记
                        cur_new_labels.append(cur_labels[audio_token_start:audio_token_start+1])
                        # 更新新的标签列表
                        cur_labels = cur_labels[audio_token_start+2:]
                else:
                    # 直接在标记处取作为现在的嵌入
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:audio_token_start]))
                    cur_new_input_embeds.append(cur_audio_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:audio_token_start])
                        cur_new_labels.append(torch.full((cur_audio_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[audio_token_start+1:]
                cur_audio_idx += 1
                # 没有用 这些标记就手动累加
                if getattr(self.config, 'tune_mm_audio_mlp_adapter', False) and getattr(self.config, 'mm_audio_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[audio_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[audio_token_start+1:]
                audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_audio_mlp_adapter', False) and getattr(self.config, 'mm_audio_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # 图像音频多模态label加载
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, audio
    ):
        # 视觉头
        vision_tower = self.get_vision_tower()

        # 音频头
        audio_tower = self.get_audio_tower()
        
        # 检查视觉头、图像输入或文本输入是否丢失    后加入音频头和音频是否丢失
        if audio_tower is None or audio is None or input_ids.shape[1] == 1 or vision_tower is None or images is None:
            # 如果文本输入只有 1 个标记  视觉头和图片有  并且给出了过去的键值
            if past_key_values is not None and audio_tower is not None and audio is not None and input_ids.shape[1] == 1:
                # 扩展注意力掩码以包含过去的键值长度
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)

            # 返回输入 ID、注意掩码、过去的键值、无图像特征、无音频特征、标签
            return input_ids, attention_mask, past_key_values, None, labels

        # 判断音频输入是列表还是一个
        # 检查audio维度是否5维，确定是否批量输入
        # 将一个或一列表输入的图像转换成特征图
        if (type(audio) is list and type(images) is list)or (audio.ndim == 5 and images.ndim == 5):
            # 将所有音频连接成一个张量
            concat_audio = torch.cat([aud for aud in audio], dim=0)
            # 对串联的音频进行编码
            audio_features = self.encode_audio(concat_audio)
            # 获取每个音频连接之前的尺寸
            split_sizes = [aud.shape[0] for aud in audio]
            # 将编码特征拆分回原批次
            audio_features = torch.split(audio_features, split_sizes, dim=0)
            # 最后将编码特征展平
            audio_features = [x.flatten(0, 1) for x in audio_features]

            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            audio_features = self.encode_audio(audio)
            image_features = self.encode_images(images)

        # Concatenate along a new features dimension
        concat_features = []
        if audio_features is not None:
            concat_features.append(audio_features)
        if image_features is not None:
            concat_features.append(image_features)
            
        concat_features = torch.cat(concat_features, dim=1)

        # Flatten to single vector per sample
        final_features = concat_features.flatten(start_dim=0, end_dim=1)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_img_audio_idx = 0
        # 遍历每个输入
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 查看是否包含特殊总体标记 TOKEN
            if (cur_input_ids == COMPACT_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # 无图，无音频，只有文字 则将文字作为嵌入
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_audio_projector(audio_tower.dummy_feature)).sum() + (0. * self.get_model().mm_vision_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                # 如果没有，则将文本输入附加到new_input_embeds
                # 将该id与label对其
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_img_audio_idx += 1
                continue

            # 找到综合切片位置
            compact_token_indices = torch.where(cur_input_ids == COMPACT_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # 音频位置切片遍历
            while compact_token_indices.numel() > 0:
                # 从合并的final_features列表中获取对应的合并特征
                cur_compact_features = final_features[cur_img_audio_idx]
                compact_token_start = compact_token_indices[0]
                # 在综合标记索引处拆分输入 ID 和标签
                if getattr(self.config, 'tune_mm_audio_mlp_adapter', False) and getattr(self.config, 'mm_audio_use_im_start_end', False) and getattr(self.config, 'tune_mm_vision_mlp_adapter', False) and getattr(self.config, 'mm_vision_use_im_start_end', False):
                    # 可以选择在综合特征周围添加开始/结束标记
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:compact_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[compact_token_start-1:compact_token_start]))
                    # 将综合特征附加到嵌入中
                    cur_new_input_embeds.append(cur_compact_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[compact_token_start+1:compact_token_start+2]))

                    if labels is not None:
                        # 在综合索引之前和之后附加文本嵌入
                        cur_new_labels.append(cur_labels[:compact_token_start])
                        # 将综合区域标签设置为 IGNORE_INDEX
                        cur_new_labels.append(torch.full((cur_compact_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # 嵌入任何剩余的文本标记
                        cur_new_labels.append(cur_labels[compact_token_start:compact_token_start+1])
                        # 更新新的标签列表
                        cur_labels = cur_labels[compact_token_start+2:]
                else:
                    # 直接在标记处取作为现在的嵌入
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:compact_token_start]))
                    cur_new_input_embeds.append(cur_compact_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:compact_token_start])
                        cur_new_labels.append(torch.full((cur_compact_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[compact_token_start+1:]
                cur_img_audio_idx += 1
                # 没有用标志头 这些标记就手动累加
                if getattr(self.config, 'tune_mm_audio_mlp_adapter', False) and getattr(self.config, 'mm_audio_use_im_start_end', False) and getattr(self.config, 'tune_mm_vision_mlp_adapter', False) and getattr(self.config, 'mm_vision_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[compact_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[compact_token_start+1:]
                audio_token_indices = torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_audio_mlp_adapter', False) and getattr(self.config, 'mm_audio_use_im_start_end', False) and getattr(self.config, 'tune_mm_vision_mlp_adapter', False) and getattr(self.config, 'mm_vision_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # 视觉token初始化
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_vision_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        # 图像起始结束位置token
        if model_args.mm_vision_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # tokenize之后有数据
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 图像微调层 adapter layer
            if model_args.tune_mm_vision_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # 图像投影层预训练
            if model_args.pretrain_mm_vision_mlp_adapter:
                mm_vision_projector_weights = torch.load(model_args.pretrain_mm_vision_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_vision_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        # 图像补丁
        elif model_args.mm_vision_use_im_patch_token:
            if model_args.tune_mm_vision_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


    # 音频token初始化
    def initialize_audio_tokenizer(self, model_args, tokenizer):
        if model_args.mm_audio_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        # 音频起始结束位置token
        if model_args.mm_audio_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_AU_START_TOKEN, DEFAULT_AU_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # tokenize之后有数据
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # 音频微调层 adapter layer
            if model_args.tune_mm_audio_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            # 音频投影层预训练
            if model_args.pretrain_mm_audio_mlp_adapter:
                mm_audio_projector_weights = torch.load(model_args.pretrain_mm_audio_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_audio_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        # 音频补丁
        elif model_args.mm_audio_use_im_patch_token:
            if model_args.tune_mm_audio_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False