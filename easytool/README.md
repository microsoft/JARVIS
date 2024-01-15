<p align="center">
<img src="./assets/logo.png" width="15%"> <br>
</p>

<div align="center">
<h1>EasyTool</h1>
<h3>Enhancing LLM-based Agents with Concise Tool Instruction<h3>
</div>

## What's New
+  [2024.01.15] We release Easytool for easier tool usage.
   + The code and datasets are available at [easytool](#).
   + The paper is available at [EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction](https://arxiv.org/abs/2401.06201).
   
## Overview

LLM-based agents usually employ tool documentation to grasp the selection and usage of tools from different sources, but these documentations could be inconsistent in formats, redundant with excessive length, and lacking demonstrations for instructions. 

EasyTool is an easy but effective method to create clear, structured, and unified instructions from tool documentations for improving LLM-based agents in using tools.

<p align="center">
<img width="70%" alt="image" src="./assets/front.png">    
</p>

## Experiment

### Prerequisites

- Prepare requirements: `pip install -r requirements.txt`
- Data Construction: `python3 data_process.py`
  
Before running any of the commands, ensure that you have set the necessary API keys. Replace `""` with your actual keys.
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export RAPIDAPI_KEY="your_rapidapi_key_here"
```
### ToolBench
You need first get the tool execution code (./data/toolenv/tools.) from the following link: [Google Drive](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/) and then save them to ./toolenv/tools
To inference with LLMs, run the following commands:
```bash
unzip data_toolbench/tool_instruction/API_description_embeddings.zip -d data_toolbench/tool_instruction/

export OPENAI_API_KEY=""
export RAPIDAPI_KEY=""

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task toolbench \
    --data_type G2 \
    --tool_root_dir ./toolenv/tools

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task toolbench \
    --data_type G3 \
    --tool_root_dir ./toolenv/tools

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task toolbench_retrieve \
    --data_type G2 \
    --tool_root_dir ./toolenv/tools

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task toolbench_retrieve \
    --data_type G3 \
    --tool_root_dir ./toolenv/tools
```

### FuncQA

To inference with LLMs, run the following commands:
```bash
export OPENAI_API_KEY=""

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task funcqa \
    --data_type funcqa_mh

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task funcqa \
    --data_type funcqa_oh
```

### RestBench

To inference with LLMs, run the following commands:
```bash
export OPENAI_API_KEY=""

python3 main.py \
    --model_name gpt-3.5-turbo \
    --task restbench 
```

## Citation

If you find this work useful in your method, you can cite the paper as below:

    @article{yuan2024easytool,
      title   = {EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction}, 
      author  = {Siyu Yuan and Kaitao Song and Jiangjie Chen and Xu Tan and Yongliang Shen and Ren Kan and Dongsheng Li and Deqing Yang},
      journal = {arXiv preprint arXiv:2401.06201},
      year    = {2024}
    }

## Acknowledgement

- [ChatGPT](https://platform.openai.com/)
- [Hugging Face](https://huggingface.co/)
- [ToolBench](https://github.com/OpenBMB/ToolBench)
- [RestBench](https://github.com/Yifan-Song793/RestGPT)
- [FuncQA](https://github.com/Ber666/ToolkenGPT)
