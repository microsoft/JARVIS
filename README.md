# JARVIS

**This project is under construction and we will have all the code ready soon.**

## Overview

Language serves as an interface for LLMs to connect numerous AI models for solving complicated AI tasks!

See our paper: [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](http://arxiv.org/abs/2303.17580)

<p align="center"><img src="./assets/overview.jpg"></p>

**HuggingGPT** is a collaborative system that consists of **an LLM as the controller** and **numerous expert models as collaborative executors** (from HuggingFace Hub). The workflow of HuggingGPT consists of four stages:
+ **Task Planning**: Using ChatGPT to analyze the requests of users to understand their intention, and disassemble them into possible solvable sub-tasks.
+ **Model Selection**: Based on the sub-tasks, ChatGPT invoke the corresponding models hosted on HuggingFace.
+ **Task Execution**: Executing each invoked model and returning the results to ChatGPT.
+ **Response Generation**: Finally, using ChatGPT to integrate the prediction of all models, and generate response.

## Quick Start

First replace `openai.key` in `server/config.yaml` with your personal key. Then run the following commands:

For server:

```bash
# setup env
cd server
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch 
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt

# download models
cd models
sh download.sh

# run server
cd ..
python bot_server.py
python model_server.py
```

For web:

```bash
cd web
npm install
npm run dev
```

Work in progress...

## Screenshots

<p align="center"><img src="./assets/screenshot_q.jpg"><img src="./assets/screenshot_a.jpg"></p>

## Citation
You can cite HuggingGPT as follows:

    @article{shen2023hugginggpt,
        title={HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace},
        author={Shen, Yongliang and Song, Kaitao and Tan, Xu and Li, Dongsheng and Lu, Weiming and Zhuang, Yueting},
        journal={arXiv preprint arXiv:2303.17580},
        year={2023}
    }

## Acknowledgement

- [ChatGPT](https://platform.openai.com/)
- [HuggingFace](https://huggingface.co/)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
