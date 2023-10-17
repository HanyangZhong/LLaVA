from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX,AUDIO_TOKEN_INDEX,COMPACT_TOKEN_INDEX
import base64
import io
from scipy.io.wavfile import read as wav_read

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def load_audio_from_base64(audio_b64):
    
    # Decode base64 string to bytes
    audio_bytes = base64.b64decode(audio_b64)
    
    # Convert bytes to BytesIO object
    audio_io = io.BytesIO(audio_bytes)

    # Load audio using scipy wavfile module
    sampling_rate, audio_array = wav_read(audio_io)

    return audio_array, sampling_rate

def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']

def process_audio(audio_array,sampling_rate, audio_processor, model_cfg):
    inputs = audio_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    return inputs

# 图像tokenizer
# "A photo of a <image> in a forest"
# "A photo of a"
# "<image>" → [image_token_index]
# "in a forest"
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # 将提示拆分为<image>上的块，以分隔文本段
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    # 使用提供的分词器将每个文本块分词为 id
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 在每个文本块之间插入给定的 image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# 音频tokenizer
# "A audio of a <audio> in a forest"
# "A audio of a"
# "<audio>" → [audio_token_index]
# "in a forest"
def tokenizer_audio_token(prompt, tokenizer, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<audio>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [audio_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# 图像音频tokenizer
# "<image><audio>please describe the image and audio in compact"
# "<image><audio>" → [compact_token_index]
# "please describe the image and audio in compact"
def tokenizer_compact_token(prompt, tokenizer, compact_token_index=COMPACT_TOKEN_INDEX, return_tensors=None):
    # 将提示拆分为<image>和<audio>上的块，以分隔文本段
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image><audio>')]

    # 使用提供的分词器将每个文本块分词为 id
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # Insert both image and audio tokens between chunks
    for x in insert_separator(prompt_chunks, [compact_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
