# JARVIS

**This project is under construction and we will have all the code ready soon.**

## Updates

+  [2023.4.1] We update a version of code for building.

## Overview

Language serves as an interface for LLMs to connect numerous AI models for solving complicated AI tasks!

See our paper: [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](http://arxiv.org/abs/2303.17580)

<p align="center"><img src="./assets/overview.jpg"></p>

We introduce a collaborative system that consists of **an LLM as the controller** and **numerous expert models as collaborative executors** (from HuggingFace Hub). The workflow of our system consists of four stages:
+ **Task Planning**: Using ChatGPT to analyze the requests of users to understand their intention, and disassemble them into possible solvable sub-tasks.
+ **Model Selection**: Based on the sub-tasks, ChatGPT invoke the corresponding models hosted on HuggingFace.
+ **Task Execution**: Executing each invoked model and returning the results to ChatGPT.
+ **Response Generation**: Finally, using ChatGPT to integrate the prediction of all models, and generate response.

## System Requirements

+ Ubuntu 20.04 LTS
+ NVIDIA GeForce RTX 3090 * 1
+ RAM >= 80GB

## Quick Start

First replace `openai.key` and `huggingface.cookie` in `server/config.yaml` with **your personal key** and **your cookies at huggingface.co**. Then run the following commands:

For server:

```bash
# setup env
cd server
conda create -n jarvis python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

# download models
cd models
sh download.sh

# run server
cd ..
python models_server.py
python bot_server.py --config config.yaml # for text-davinci-003
```

For web:

```bash
cd web
npm install
npm run dev
```

Note that in order to display the video properly in HTML, you need to compile `ffmpeg` manually with H.264

```bash
# This command need be executed without errors.
LD_LIBRARY_PATH=/usr/local/lib /usr/local/bin/ffmpeg -i input.mp4 -vcodec libx264 output.mp4
```

## Screenshots

<p align="center"><img src="./assets/screenshot_q.jpg"><img src="./assets/screenshot_a.jpg"></p>

## Citation
If you find this work useful in your method, you can cite the paper as below:

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
- [ChatGPT-vue](https://github.com/lianginx/chatgpt-vue)