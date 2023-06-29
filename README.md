# 🌋 LLaVA: Large Language and Vision Assistant

*Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

[[Project Page](https://llava-vl.github.io/)] [[Paper](https://arxiv.org/abs/2304.08485)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)] [[Model](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0)]

**Visual Instruction Tuning** <br>
[Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)

<p align="center">
    <a href="https://llava.hliu.cc/"><img src="images/llava_logo.png" width="50%"></a> <br>
    Generated by <a href="https://gligen.github.io/">GLIGEN</a> via "a cute lava llama with glasses" and box prompt
</p>


## Release
- [6/26] 🔥 [CVPR 2023 Tutorial](https://vlp-tutorial.github.io/) on **Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4**!  Please check out [[Slides](https://datarelease.blob.core.windows.net/tutorial/vision_foundation_models_2023/slides/Chunyuan_cvpr2023_tutorial_lmm.pdf)] [[Notes](https://arxiv.org/abs/2306.14895)] [[YouTube](https://youtu.be/mkI7EPD1vp8)] [[Bilibli](https://www.bilibili.com/video/BV1Ng4y1T7v3/)].
- [6/11] 🔥 We released the preview for the mostly requested feature: DeepSpeed and LoRA support!  Please see documentations [here](./docs/LoRA.md).
- [6/1] 🔥 We released **LLaVA-Med: Large Language and Vision Assistant for Biomedicine**, a step towards building biomedical domain large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2306.00890) and [page](https://github.com/microsoft/LLaVA-Med).
- [5/13] 🔥 Interested in quantifying the emerged **zero-shot OCR** performance of LLaVA and open-sourced LMM? Please check out the paper ["On the Hidden Mystery of OCR in Large Multimodal Models"](https://arxiv.org/abs/2305.07895), where LLaVA consistently outperforms miniGPT4 on 17 out of 18 datasets, despite LlaVA being trained with an order of magnitude smaller training data.
- [5/6] 🔥 We are releasing [LLaVA-Lighting-MPT-7B-preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview), based on MPT-7B-Chat!  See [here](#LLaVA-MPT-7b) for more details.
- [5/2] 🔥 We are releasing LLaVA-Lighting!  Train a lite, multimodal GPT-4 with just $40 in 3 hours!  See [here](#train-llava-lightning) for more details.
- [5/2] We upgrade LLaVA package to v0.1 to support Vicuna v0 and v1 checkpoints, please upgrade following instructions [here](#install).
- [4/30] Our checkpoint with Vicuna-7b-v0 has been released [here](#llava-7b)! This checkpoint is more accessible and device friendly.  Stay tuned for a major upgrade next week!
- [4/27] Thanks to the community effort, LLaVA-13B with 4-bit quantization allows you to run on a GPU with as few as 12GB VRAM!  Try it out [here](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llava).
- [4/17] 🔥 We released **LLaVA: Large Language and Vision Assistant**. We propose visual instruction tuning, towards building large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2304.08485) and [demo](https://llava.hliu.cc/).

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Contents
- [Install](#install)
- [LLaVA Weights](#llava-weights)
- [Demo](#Demo)
- [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn==1.0.2
```

### Upgrade to latest code base

```Shell
git pull
pip uninstall transformers
pip install -e .
```

## LLaVA Weights
We release [LLaVA](https://llava-vl.github.io/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the LLaVA weights.

Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get LLaVA weights by applying our delta ([13b-v0](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0), [7b-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0), [lightning-7B-v1-1](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1)). It will automatically download delta weights from our Hugging Face account.

```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/LLaVA-7B-v0 \
    --delta liuhaotian/LLaVA-7b-delta-v0
```

## Demo

To run our demo, you need to prepare LLaVA checkpoints locally.  Please follow the instructions [here](#llava-weights) to download the checkpoints.

### Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server *ONCE*.

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```
You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker

This is the actual *worker* that performs the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/LLaVA-13B-v0
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.
```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 40000, say 40001> --worker http://localhost:<change accordingly, i.e. 40001> --model-path <ckpt2>
```

#### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If your the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/LLaVA-13B-v0 --num-gpus 2
```

### CLI Inference

A starting script for inference with LLaVA without the need of Gradio interface. The current implementation only supports for a single-turn Q-A session, and the interactive CLI is WIP.  This also serves as an example for users to build customized inference scripts.

```Shell
python -m llava.eval.run_llava \
    --model-name /path/to/LLaVA-13B-v0 \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --query "What are the things I should be cautious about when I visit here?"
```

Example output (varies in different runs):

> When visiting this picturesque location with a serene lake and a wooden pier extending over the water, one should be cautious about various safety aspects. Some important considerations include:
> 
> 1. Ensuring that the pier is structurally sound andstable, as old or weakened pier structures might not support the weight of visitors.
> 2. Being aware of the water depth around the pier and lake, as sudden drop-offs or strong currents may pose a risk to swimmers, boaters, or those who venture too close to the edge.
> 3. Staying vigilant about the presence of wildlife in the area, such as slippery, stealthy fish or other animals that might cause harm or inconvenience.
> 4. Maintaining a safe distance from the water's edge, particularly for children, elderly individuals, or those who are not strong swimmers.
> 5. Following any posted signs or guidelines related to safety and the use of the pier and surrounding areas.
> 
> By considering these safety precautions, visitors can enjoy the natural beauty of the location while minimizing risks and ensuring a safe and pleasant experience.

## Train

LLaVA training consists of two stages: (1) feature alignment stage: use approximately 600K filtered CC3M to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following to teach the model to follow multimodal instructions.

LLaVA is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps`.

### Hyperparameters
We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 128 | 2e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 32 | 2e-5 | 3 | 2048 | 0 |

### Prepare Vicuna checkpoints

Before you start, prepare our base model Vicuna, which is an instruction-tuned chatbot. Please download its weights [here](https://github.com/lm-sys/FastChat#model-weights).

Vicuna has two versions: v0 and v1, the main difference between them is the prompt of format. We support both. To ensure the best performance, you need to specify the correct prompt version corresponding to the weights you download: `v0` for `v0` weights, and `v1` for all Vicuna `v1.x` models.

### Pretrain (feature alignment)

Please download the subset of the CC3M dataset we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K).

Pretrain takes around 4 hours for LLaVA-13B on 8x A100 (80G). It takes around 2 hours for 7B checkpoints.

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/vicuna-13b \
    --version [v0 or v1] \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

You may run this with a single A100 GPU with the following code.  Please note that the `per_device_train_batch_size` * `gradient_accumulation_steps` should be equal to 128 to keep the global batch size the same.

<details>
<summary>Pretrain: LLaVA-13B, 1x A100 (80G).  Time: ~33 hours.</summary>

```Shell
python llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/vicuna-13b \
    --version [v0 or v1] \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>

<details>
<summary>Pretrain: LLaVA-7B, 1x A100 (80G/40G).  Time: ~19 hours.</summary>

```Shell
python llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/vicuna-7b \
    --version [v0 or v1] \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>


### Visual Instruction Tuning

1. Prepare data

Please download the annotation of our instruction tuning data [llava_instruct_158k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json), and download the COCO train2017 images [here](https://cocodataset.org/#download).

2. Extract projector features from the pretrained model from the feature alignment stage.

```Shell
python scripts/extract_mm_projector.py \
  --model_name_or_path ./checkpoints/llava-13b-pretrain \
  --output ./checkpoints/mm_projector/llava-13b-pretrain.bin
```

3. Start training!

You may download our pretrained `llava-13b-pretrain.bin` [here](https://huggingface.co/liuhaotian/LLaVA-Pretrained-Projectors/blob/main/LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption.bin).

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/vicuna-13b \
    --version [v0 or v1] \
    --data_path ./playground/data/llava_instruct_158k.json \
    --image_folder /path/to/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/llava-13b-pretrain.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```
</details>

### Lightning

*NOTE: When comparing to LLaVA-Lightning checkpoints in the paper, please use `LLaVA (Lightning)` instead of `LLaVA` as they use different set of training data and schedule.*

LLaVA-Lightning can be trained on 8x A100 GPUs in just 3 hours, including both pretraining and finetuning. When using spot instances, it costs just ~$40.

For LLaVA Lightning, we create two distilled subset to ensure both a broad concept coverage, and the efficiency in training. Furthermore, we only perform instruction tuning for 1 epoch, in contrast to 3 epochs in the paper.

For pretraining, we create a concept-balanced subset of LAION-CC-SBU. It consists of 558K images.  Download data [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main).

For instruction tuning, we create a subset of LLaVA-Instruct-150K. It consists of 80K image-instruction pairs, consisting of 40K conversation and 40K complex reasoning data, with non-overlapping images. Download `llava_instruct_80k.json` [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_80k.json).


```Shell
bash ./scripts/train_lightning.sh {v0,v1}
```

#### Hyperparameters

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-Lightning-7B | 128 | 2e-3 | 1 | 2048 | 0 |

2. Visual Instruction Tuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-Lightning-7B | 128 | 2e-5 | 1 | 2048 | 0 |

#### LLaVA-MPT-7b
Thanks to LLaVA-Lightning, we are able to train a checkpoint based on MPT-7b-Chat on 8x A100 GPUs in just 3 hours, including both pretraining and finetuning.

*NOTE: When comparing to LLaVA-MPT-7B checkpoints in the paper, please use `LLaVA-MPT-7B (Lightning)` instead of `LLaVA` as they use different set of base LLM, training data and schedule.*

**NOTE**: This is a research preview of the LLaVA-Lightning based on MPT-7B-chat checkpoint. The usage of the model should comply with MPT-7B-chat license and agreements.

**NOTE**: Unlike other LLaVA models, this model should be used directly without delta weights conversion!

**NOTE**: You need to upgrade to our latest code base to use LLaVA-MPT-7b!

1. Usage

You do not need to download our checkpoint, it will directly load from our Hugging Face model: [`liuhaotian/LLaVA-Lightning-MPT-7B-preview`](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview).

```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/LLaVA-Lightning-MPT-7B-preview
python -m llava.serve.gradio_web_server --controller http://localhost:10000
```

2. Training

We use the same set of training dataset, and the hyperparameters as other Lightning checkpoints.

```Shell
bash ./scripts/train_lightning_mpt.sh
```

### ScienceQA
**NOTE**: Due to that ScienceQA experiments were done earlier, the current checkpoints are trained *without* `<im_start>` and `<im_end>` tokens. Here we provide our training scripts for the current checkpoints.

<details>
<summary>1. Pretraining</summary>

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llama-vicuna-13b \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain-no_im_start_end_token \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>

<details>
<summary>2. Extract projector features</summary>

```Shell
python scripts/extract_mm_projector.py \
  --model_name_or_path ./checkpoints/llava-13b-pretrain-no_im_start_end_token \
  --output ./checkpoints/mm_projector/llava-13b-pretrain-no_im_start_end_token.bin
```
</details>

<details>
<summary>3. Finetuning</summary>

You may download our pretrained `llava-13b-pretrain-no_im_start_end_token.bin` [here](https://huggingface.co/liuhaotian/LLaVA-13b-pretrain-projector-v0/blob/main/LLaVA-13b-pretrain-projector-v0-CC3M-595K-original_caption-no_im_token.bin).

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/llama-vicuna-13b \
    --data_path /path/to/scienceqa/llava_train_QCM-LEPA.json \
    --image_folder /path/to/scienceqa/images/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/llava-13b-pretrain-no_im_start_end_token.bin \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain-no_im_start_end_token-finetune_scienceqa \
    --num_train_epochs 12 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>

## Evaluation

### GPT-assisted Evaluation

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see our paper for more details.

1. Generate LLaVA responses

```Shell
python model_vqa.py \
    --model-name ./checkpoints/LLaVA-13B-v0 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /path/to/answer-file.jsonl
```

2. Evaluate the generated responses.  In our case, [`answer-file-1.jsonl`](./playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl) is the response generated by text-only GPT-4 (0314), with the context captions/boxes provided.

```Shell
OPENAI_API_KEY="sk-***********************************" python eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    /path/to/answer-file-1.jsonl \
    /path/to/answer-file-2.jsonl \
    --rule table/rule.json \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```

### ScienceQA

#### Prepare Data
1. Please see ScienceQA [repo](https://github.com/lupantech/ScienceQA) for setting up the dataset.
2. Generate ScienceQA dataset for LLaVA conversation-style format.

```Shell
python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --split {train,val,minival,test,minitest}
```

#### Evaluation

1. Download our pretrained LLaVA-13B (delta) weights for ScienceQA dataset [here](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0-science_qa).  Convert the delta weights to actual weights.

```Shell
python -m llava.model.apply_delta \
    --base /path/to/llama-13b \
    --target /path/to/LLaVA-13b-v0-science_qa \
    --delta liuhaotian/LLaVA-13b-delta-v0-science_qa
```

2. [Option 1] Multiple-GPU inference
You may evaluate this with multiple GPUs, and concatenate the generated jsonl files.  Please refer to our script for [batch evaluation](scripts/sqa_eval_batch.sh) and [results gathering](scripts/sqa_eval_gather.sh).

3. [Option 2] Single-GPU inference

(a) Generate LLaVA responses on ScienceQA dataset

```Shell
python -m llava.eval.model_vqa_science \
    --model-name /path/to/LLaVA-13b-v0-science_qa \
    --question-file /path/to/ScienceQA/data/scienceqa/llava_test.json \
    --image-folder /path/to/ScienceQA/data/scienceqa/images/test \
    --answers-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --answer-prompter
    --conv-mode simple
```

(b) Evaluate the generated responses

```Shell
python eval_science_qa.py \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --result-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --output-file vqa/results/ScienceQA/test_llava-13b_output.json \
    --output-result vqa/results/ScienceQA/test_llava-13b_result.json \
```

For reference, we attach our prediction file `test_llava-13b_result.json` [here](llava/eval/table/results/test_sqa_llava_13b_v0.json) for comparison when reproducing our results, as well as for further analysis in detail.

## Citation

If you find LLaVA useful for your your research and applications, please cite using this BibTeX:
```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, pleae check out:
- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
