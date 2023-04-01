models="
nlpconnect/vit-gpt2-image-captioning
lllyasviel/ControlNet
runwayml/stable-diffusion-v1-5
CompVis/stable-diffusion-v1-4
stabilityai/stable-diffusion-2-1
Salesforce/blip-image-captioning-large
damo-vilab/text-to-video-ms-1.7b
microsoft/speecht5_asr
facebook/maskformer-swin-large-ade
microsoft/biogpt
facebook/esm2_t12_35M_UR50D
microsoft/trocr-base-printed
microsoft/trocr-base-handwritten
JorisCos/DCCRNet_Libri1Mix_enhsingle_16k
espnet/kan-bayashi_ljspeech_vits
facebook/detr-resnet-101
microsoft/speecht5_tts
microsoft/speecht5_hifigan
microsoft/speecht5_vc
facebook/timesformer-base-finetuned-k400
runwayml/stable-diffusion-v1-5
superb/wav2vec2-base-superb-ks
openai/whisper-base
Intel/dpt-large
microsoft/beit-base-patch16-224-pt22k-ft22k
facebook/detr-resnet-50-panoptic
facebook/detr-resnet-50
openai/clip-vit-large-patch14
google/owlvit-base-patch32
microsoft/DialoGPT-medium
bert-base-uncased
Jean-Baptiste/camembert-ner
deepset/roberta-base-squad2
facebook/bart-large-cnn
google/tapas-base-finetuned-wtq
distilbert-base-uncased-finetuned-sst-2-english
gpt2
mrm8488/t5-base-finetuned-question-generation-ap
Jean-Baptiste/camembert-ner
t5-base
impira/layoutlm-document-qa
ydshieh/vit-gpt2-coco-en
dandelin/vilt-b32-finetuned-vqa
lambdalabs/sd-image-variations-diffusers
facebook/timesformer-base-finetuned-k400
facebook/maskformer-swin-base-coco
Intel/dpt-hybrid-midas
"

# CURRENT_DIR=$(cd `dirname $0`; pwd)
CURRENT_DIR=$(pwd)
for model in $models;
do
    echo "----- Downloading from https://huggingface.co/"$model" -----"
    if [ -d "$model" ]; then
        # cd $model && git reset --hard && git pull && git lfs pull
        cd $model && git pull && git lfs pull
        cd $CURRENT_DIR
    else
        # git clone 包含了lfs
        git clone https://huggingface.co/$model $model
    fi
done

datasets="Matthijs/cmu-arctic-xvectors"

for dataset in $datasets;
 do
     echo "----- Downloading from https://huggingface.co/datasets/"$dataset" -----"
     if [ -d "$dataset" ]; then
         cd $dataset && git pull && git lfs pull
         cd $CURRENT_DIR
     else
         git clone https://huggingface.co/datasets/$dataset $dataset
     fi
done