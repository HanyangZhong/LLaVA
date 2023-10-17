import os
import torch

from transformers import Trainer
from typing import Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

# llava训练器
class robotGPTTrainer(Trainer):

    # 保存训练中的权重
    def _save_checkpoint(self, model, trial, metrics=None):
        # 判断是否视觉头加adapte layer
        if getattr(self.args, 'tune_mm_vision_mlp_adapter', False) and (getattr(self.args, 'tune_mm_audio_mlp_adapter', True) or getattr(self.args, 'tune_mm_audio_mlp_adapter', None)):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_vision_projector']
            if getattr(self.args, "vision_use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_vision_projector.bin'))

        # 判断是否视觉头加adapte layer
        elif getattr(self.args, 'tune_mm_audio_mlp_adapter', False) and (getattr(self.args, 'tune_mm_vision_mlp_adapter', True)or getattr(self.args, 'tune_mm_vision_mlp_adapter', None)):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_audio_projector']
            if getattr(self.args, "audio_use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_audio_projector.bin'))

        # 判断是否视觉头加adapte layer
        elif getattr(self.args, 'tune_mm_audio_mlp_adapter', False) and getattr(self.args, 'tune_mm_vision_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_audio_projector']

            if getattr(self.args, "audio_use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save_a = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            # Only save Adapter
            keys_to_match = ['mm_vision_projector']
            if getattr(self.args, "vision_use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save_v = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save_a, os.path.join(output_dir, f'mm_audio_projector.bin'))
                torch.save(weight_to_save_v, os.path.join(output_dir, f'mm_vision_projector.bin'))

        else:
            super(robotGPTTrainer, self)._save_checkpoint(model, trial, metrics)


        # 判断是否综合头加adapte layer 分别在视觉和音频加
        if getattr(self.args, 'tune_mm_compact_mlp_adapter', False):
            # 音频
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match_a = ['mm_audio_projector']
            if getattr(self.args, "audio_use_im_start_end", False):
                keys_to_match_a.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match_a)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_audio_projector.bin'))

            # 视觉
            # Only save Adapter
            keys_to_match_v = ['mm_vision_projector']
            if getattr(self.args, "vision_use_im_start_end", False):
                keys_to_match_v.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match_v)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_vision_projector.bin'))
            
        else:
            super(robotGPTTrainer, self)._save_checkpoint(model, trial, metrics)

    # 保存最后的权重
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_vision_mlp_adapter', False):
            pass
        elif getattr(self.args, 'tune_mm_audio_mlp_adapter', False):
            pass
        elif getattr(self.args, 'tune_mm_compact_mlp_adapter', False):
            pass
        else:
            super(robotGPTTrainer, self)._save(output_dir, state_dict)
        
        # if getattr(self.args, 'tune_mm_audio_mlp_adapter', False):
        #     pass
        # else:
        #     super(robotGPTTrainer, self)._save(output_dir, state_dict)

        # if getattr(self.args, 'tune_mm_compact_mlp_adapter', False):
        #     pass
        # else:
        #     super(robotGPTTrainer, self)._save(output_dir, state_dict)
