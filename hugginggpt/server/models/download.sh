#!/bin/bash

# Set models and datasets to download
models=(
    "nlpconnect/vit-gpt2-image-captioning"
    "lllyasviel/ControlNet"
    "lllyasviel/sd-controlnet-canny"
    "lllyasviel/sd-controlnet-depth"
    "lllyasviel/sd-controlnet-hed"
    "lllyasviel/sd-controlnet-mlsd"
    "lllyasviel/sd-controlnet-openpose"
    "lllyasviel/sd-controlnet-scribble"
    "lllyasviel/sd-controlnet-seg"
    "runwayml/stable-diffusion-v1-5"
    "damo-vilab/text-to-video-ms-1.7b"
    "microsoft/speecht5_asr"
    "JorisCos/DCCRNet_Libri1Mix_enhsingle_16k"
    "espnet/kan-bayashi_ljspeech_vits"
    "facebook/detr-resnet-101"
    "microsoft/speecht5_hifigan"
    "microsoft/speecht5_vc"
    "openai/whisper-base"
    "Intel/dpt-large"
    "facebook/detr-resnet-50-panoptic"
    "facebook/detr-resnet-50"
    "google/owlvit-base-patch32"
    "impira/layoutlm-document-qa"
    "ydshieh/vit-gpt2-coco-en"
    "dandelin/vilt-b32-finetuned-vqa"
    "lambdalabs/sd-image-variations-diffusers"
    "facebook/maskformer-swin-base-coco"
    "Intel/dpt-hybrid-midas"
)
datasets=("Matthijs/cmu-arctic-xvectors")

# Set the current directory
CURRENT_DIR=$(pwd)

# Download models
for model in "${models[@]}"; do
	echo "----- Downloading from https://huggingface.co/${model} -----"
	if [ -d "${model}" ]; then
		(cd "${model}" && git pull && git lfs pull)
	else
		git clone --recurse-submodules "https://huggingface.co/${model}" "${model}"
	fi
done

# Download datasets
for dataset in "${datasets[@]}"; do
	echo "----- Downloading from https://huggingface.co/datasets/${dataset} -----"
	if [ -d "${dataset}" ]; then
		(cd "${dataset}" && git pull && git lfs pull)
	else
		git clone --recurse-submodules "https://huggingface.co/datasets/${dataset}" "${dataset}"
	fi
done