import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,AUDIO_TOKEN_INDEX, DEFAULT_AUDIO_TOKEN, DEFAULT_AU_START_TOKEN, DEFAULT_AU_END_TOKEN,COMPACT_TOKEN_INDEX, DEFAULT_COMPACT_TOKEN, DEFAULT_COM_START_TOKEN, DEFAULT_COM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, tokenizer_audio_token,tokenizer_compact_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from scipy.io import wavfile
from scipy.signal import resample
import os
import numpy as np

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_audio(audio_file):
    if audio_file.startswith('http') or audio_file.startswith('https'):
        response = requests.get(audio_file)
        sample_rate, dual_audio = wavfile.read(BytesIO(response.content)) 
        audio = np.mean(dual_audio, axis=1) 
        
    else:
        sample_rate, dual_audio = wavfile.read(audio_file) 
        audio = np.mean(dual_audio, axis=1) 

    if sample_rate != 16000:
            # print('len:',len(audio))
            number_of_samples = round(len(audio) * float(16000) / sample_rate)
            # number_of_samples = round( float(16000) / sample_rate)
            # print(number_of_samples)
            audio = resample(audio, number_of_samples)
            sample_rate = 16000
    return audio


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    model_name ='robotgpt'
    tokenizer, model, image_processor,audio_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv_mode = "llava_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if args.image_file is not None:
        image = load_image(args.image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    else:
        image=None
        image_tensor=None

    if args.audio_file is not None:
        audio = load_audio(args.audio_file)
        # audio_tensor = audio_processor.preprocess(audio, return_tensors='pt').half().cuda()
        audio_tensor = audio_processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values.half().cuda()
    else:
        audio=None
        audio_tensor=None

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        flag=0
        if args.image_file is not None and args.audio_file is None:
            # first message
            if model.config.mm_vision_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
            flag=1
        elif args.audio_file is not None and args.image_file is None:
            # first message
            if model.config.mm_audio_use_im_start_end:
                inp = DEFAULT_AU_START_TOKEN + DEFAULT_AUDIO_TOKEN + DEFAULT_AU_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_AUDIO_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            audio = None
            flag=2
        elif args.image_file is not None and args.audio_file is not None:
            # first message
            if model.config.mm_compact_use_im_start_end:
                inp = DEFAULT_COM_START_TOKEN + DEFAULT_COMPACT_TOKEN + DEFAULT_COM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_COMPACT_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
            audio = None
            flag=3
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print('flag is:',flag)
        if flag==1:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if flag==2:
            input_ids = tokenizer_audio_token(prompt, tokenizer, AUDIO_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if flag==3:
            input_ids = tokenizer_compact_token(prompt, tokenizer, COMPACT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                audio = audio_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
