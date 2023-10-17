import torch
import torch.nn as nn

from transformers import Data2VecAudioModel, AutoProcessor, Data2VecAudioConfig


class audioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        # 保留一个select layer的口，暂时不用
        self.select_layer = args.mm_audio_select_layer
        self.audio_select_feature = getattr(args, 'mm_audio_select_feature', 'cls_patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Data2VecAudioConfig.from_pretrained(self.audio_tower_name)

    # facebook/data2vec-audio-base-960h
    def load_model(self):
        self.audio_processor = AutoProcessor.from_pretrained(self.audio_tower_name)
        self.audio_tower = Data2VecAudioModel.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        # 直接用，不选择特征
        # audio_features = audio_forward_outs
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        if self.audio_select_feature == 'patch':
            audio_features = audio_features[:, 1:]
        elif self.audio_select_feature == 'cls_patch':
            audio_features = audio_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.audio_select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, audio):
        # 判断是单个输入音频还是 多个以列表形式输入
        if type(audio) is list:
            audio_features = []
            for audio_pat in audio:
                audio_forward_out = self.audio_tower(audio_pat.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(audio_pat.dtype)
                audio_features.append(audio_feature)
                # 全部保存下来
        else:
            audio_forward_outs = self.audio_tower(audio.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audio.dtype)


        return audio_features

    # 它创建一个形状为 (1, self.hidden_​​size) 的张量
    # 张量用零填充   张量设备和数据类型分别设置为匹配 self.device 和 self.dtype。
    @property
    def dummy_feature(self):
<<<<<<< Updated upstream
        return torch.zeros(1, self.audio_hidden_size, device=self.device, dtype=self.dtype)
=======
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
>>>>>>> Stashed changes

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
<<<<<<< Updated upstream
    def audio_hidden_size(self):
        return self.config.audio_hidden_size
=======
    def hidden_size(self):
        return self.config.hidden_size
>>>>>>> Stashed changes

    @property
    def num_patches(self):
        return (self.config.audio_size // self.config.patch_size) ** 2
