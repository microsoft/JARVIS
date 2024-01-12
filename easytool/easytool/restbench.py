# — coding: utf-8 –
import openai
import json
import logging
import sys
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain import LLMChain
import numpy as np
import requests
import os
import subprocess
import re
import importlib.util
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from util import *

from tqdm import tqdm

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_last_processed_index(progress_file):
    """Retrieve the last processed index from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            last_index = f.read().strip()
            return int(last_index) if last_index else 0
    else:
        return 0


def update_progress(progress_file, index):
    """Update the last processed index in the progress file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        f.write(str(index))


def task_decompose(question, Tool_dic, model_name):
    chat = ChatOpenAI(model_name=model_name)
    template = "You are a helpful assistant."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "We have spotify database and the following tools:\n"
        "{Tool_dic}"
        "You need to decompose a complex user's question into some simple subtasks and let the model execute it step by step with these tools.\n"
        "Please note that: \n"
        "1. you should break down tasks into appropriate subtasks to use the tools mentioned above.\n"
        "2. You should not only list the subtask, but also list the ID of the tool used to solve this subtask.\n"
        "3. If you think you do not need to use the tool to solve the subtask, just leave it as {{\"ID\": -1}}\n"
        "4. You must consider the logical connections, order and constraints among the tools to achieve a correct tool path."
        "5. You must ONLY output the ID of the tool you chose in a parsible JSON format. Two examples output look like:\n"
        "'''\n"
        "Question: Pause the player"
        "Example 1: [{{\"Task\":\"Get information about the user’s current playback state\", \"ID\":15}}, {{\"Task\":\"Pause playback on the user's account\", \"ID\":19}}]\n"
        "'''\n"
        "This is the user's question: {question}\n"
        "Output:"
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    ind = 0
    while True:
        try:
            result = chain.run(question=question, Tool_dic=Tool_dic)
            result = eval(result.split('\n\n')[0])
            break
        except Exception as e:
            print(f"task decompose fails: {e}")
            if ind > 10:
                return -1
            ind += 1
            continue
    return result


def task_execution(
        Tool_dic, dic_tool, test_data, progress_file,
        start_index, total_files, retrieval_num, ind, model_name):
    with tqdm(total=total_files, desc="Processing files", initial=start_index) as pbar:
        for i, data in enumerate(test_data[start_index:], start=start_index):
            question = data["query"]
            print(question)
            task_path = task_decompose(question, Tool_dic, model_name)
            tool_choice_ls = []
            for task in task_path:
                if isinstance(task["ID"], list):
                    for ele in task["ID"]:
                        tool_choice_ls.append(dic_tool[ele]['tool_usage'])
                elif int(task["ID"]) in dic_tool.keys():
                    tool_choice_ls.append(dic_tool[task["ID"]]['tool_usage'])
            ind = ind + 1
            with open(f"restbench_{model_name}_Easytool.jsonl", 'a+', encoding='utf-8') as f:
                line = json.dumps({
                    "ID": ind,
                    "question": question,
                    "task_path": task_path,
                    "tool_choice_ls": tool_choice_ls
                }, ensure_ascii=False)
                f.write(line + '\n')
            print(tool_choice_ls)
            update_progress(progress_file, i + 1)
            pbar.update(1)
