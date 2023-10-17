import os
from .audio_encoder import audioTower

# 用data2Vec做音频头
# 创建音频头
def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    # 判断后加载
    if is_absolute_path_exists or audio_tower.startswith("facebook"):
        print('loading audio tower')
        return audioTower(audio_tower, args=audio_tower_cfg, **kwargs)

    raise ValueError(f'Unknown audio tower: {audio_tower}')
