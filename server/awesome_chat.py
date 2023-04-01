import base64
from io import BytesIO
import random
from time import sleep
import time
import traceback
import uuid
import numpy as np
import requests
import re
import json
import logging
import argparse
import yaml
from PIL import Image, ImageDraw
from pydub import AudioSegment
import multiprocessing
from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model, count_tokens


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

LLM = config["model"]
use_completion = config["use_completion"]

# consistent: wrong msra model name 
LLM_encoding = LLM
if LLM == "gpt-3.5-turbo":
    LLM_encoding = "text-davinci-003"
task_parsing_highlight_ids = get_token_ids_for_task_parsing(LLM_encoding)
choose_model_highlight_ids = get_token_ids_for_choose_model(LLM_encoding)

# ENDPOINT	MODEL NAME	
# /v1/chat/completions	gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301	
# /v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada

if use_completion:
    api_name = "completions"
else:
    api_name = "chat/completions"

if not config["dev"] and config["openai"]["key"]:
    OPENAI_KEY = config["openai"]["key"]
    endpoint = f"https://api.openai.com/v1/{api_name}"
    HEADER = {
        "Authorization": f"Bearer {OPENAI_KEY}"
    }
else:
    endpoint = f"{config['local']['endpoint']}/v1/{api_name}"
    HEADER = None

PROXY = None
if config["proxy"]:
    PROXY = {
        "https": config["proxy"],
    }
    

HTTP_Server = "http://" + config["httpserver"]["host"] + ":" + str(config["httpserver"]["port"])
Model_Server = "http://" + config["modelserver"]["host"] + ":" + str(config["modelserver"]["port"])


parse_task_demos_or_presteps = open(config["demos_or_presteps"]["parse_task"], "r").read()
choose_model_demos_or_presteps = open(config["demos_or_presteps"]["choose_model"], "r").read()
response_results_demos_or_presteps = open(config["demos_or_presteps"]["response_results"], "r").read()


parse_task_prompt = config["prompt"]["parse_task"]
choose_model_prompt = config["prompt"]["choose_model"]
response_results_prompt = config["prompt"]["response_results"]

parse_task_tprompt = config["tprompt"]["parse_task"]
choose_model_tprompt = config["tprompt"]["choose_model"]
response_results_tprompt = config["tprompt"]["response_results"]

MODELS = [json.loads(line) for line in open("data/p0_models.jsonl", "r").readlines()]
MODELS_MAP = {}
for model in MODELS:
    tag = model["task"]
    if tag not in MODELS_MAP:
        MODELS_MAP[tag] = []
    MODELS_MAP[tag].append(model)
METADATAS = {}
for model in MODELS:
    METADATAS[model["id"]] = model

HUGGINGFACE_HEADERS = {}
if config["huggingface"]["cookie"]:
    HUGGINGFACE_HEADERS = {
        "cookie": config["huggingface"]["cookie"],
    }

def convert_chat_to_completion(data):
    messages = data.pop('messages', [])
    tprompt = ""
    if messages[0]['role'] == "system":
        tprompt = messages[0]['content']
        messages = messages[1:]
    final_prompt = ""
    for message in messages:
        if message['role'] == "user":
            final_prompt += ("<im_start>"+ "user" + "\n" + message['content'] + "<im_end>\n")
        elif message['role'] == "assistant":
            final_prompt += ("<im_start>"+ "assistant" + "\n" + message['content'] + "<im_end>\n")
        else:
            final_prompt += ("<im_start>"+ "system" + "\n" + message['content'] + "<im_end>\n")
    final_prompt = tprompt + final_prompt
    final_prompt = final_prompt + "<im_start>assistant"
    data["prompt"] = final_prompt
    data['stop'] = data.get('stop', ["<im_end>"])
    data['max_tokens'] = data.get('max_tokens', max(4000 - count_tokens(LLM_encoding, final_prompt), 1))
    return data

def send_request(data):
    if use_completion:
        data = convert_chat_to_completion(data)
    response = requests.post(endpoint, json=data, headers=HEADER, proxies=PROXY)
    if use_completion:
        return response.json()["choices"][0]["text"].replace("\n", "")
    else:
        return response.json()["choices"][0]["message"]["content"].replace("\n", "")

def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'").replace('\n', ""))
    return text


def find_json(s):
    s = s.replace("\'", "\"")
    start = s.find("{")
    end = s.rfind("}")
    res = s[start:end+1]
    res = res.replace("\n", "")
    return res

def field_extract(s, field):
    try:
        field_rep = re.compile(f'{field}.*?:.*?"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    except:
        field_rep = re.compile(f'{field}:\ *"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    return extracted

def record_case(success, **args):
    if success:
        f = open("log_success.jsonl", "a")
    else:
        f = open("log_fail.jsonl", "a")
    log = args
    f.write(json.dumps(log) + "\n")
    f.close()

def chitchat(messages):
    data = {
        "model": LLM,
        "messages": messages
    }
    return send_request(data)

def parse_task(context, input):

    demos_or_presteps = parse_task_demos_or_presteps
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": parse_task_tprompt})

    start = 0
    while start <= len(context):
        history = context[start:]
        prompt = replace_slot(parse_task_prompt, {
            "input": input,
            "context": history 
        })
        messages.append({"role": "user", "content": prompt})
        history_text = "<im_end>\nuser<im_start>".join([m["content"] for m in messages])
        num = count_tokens(LLM_encoding, history_text)
        if 4000 - num > 800:
            break
        messages.pop()
        start += 2
    
    logger.info(messages)
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0,
        "logit_bias": {item: config["logit_bias"]["parse_task"] for item in task_parsing_highlight_ids}
    }
    return send_request(data)

def choose_model(input, task, metas, preference=["High efficiency", "Good performance."]):
    prompt = replace_slot(choose_model_prompt, {
        "input": input,
        "task": task,
        "metas": metas,
    })
    demos_or_presteps = replace_slot(choose_model_demos_or_presteps, {
        "input": input,
        "task": task,
        "metas": metas
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": choose_model_tprompt})
    messages.append({"role": "user", "content": prompt})
    logger.info(messages)
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0,
        "logit_bias": {item: config["logit_bias"]["choose_model"] for item in choose_model_highlight_ids} # 5
    }
    return send_request(data)


def response_results(input, results):
    results = [v for k, v in sorted(results.items(), key=lambda item: item[0])]
    prompt = replace_slot(response_results_prompt, {
        "input": input,
    })
    demos_or_presteps = replace_slot(response_results_demos_or_presteps, {
        "input": input,
        "processes": results
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": response_results_tprompt})
    messages.append({"role": "user", "content": prompt})
    logger.info(messages)
    data = {
        "model": LLM,
        "messages": messages,
        "temperature": 0
    }
    return send_request(data)

def hugginhface_model_inference(model_id, data, task):
    task_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # NLP tasks
    if task == "question-answering":
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json={"inputs": {"question": data["text"], "context": (data["context"] if "context" in data else "" )}})
        return response.json()
    if task == "sentence-similarity":
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json={"inputs": {"source_sentence": data["text1"], "target_sentence": data["text2"]}})
        return response.json()
    if task in ["text-classification",  "token-classification", "text2text-generation", "summarization", "translation", "conversational", "text-generation"]:
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json=data)
        return response.json()
    
    # CV tasks
    if task == "visual-question-answering" or task == "document-question-answering":
        img_url = data["image"]
        text = data["text"]
        img_data = requests.get(img_url, timeout=10).content
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        json_data = {}
        json_data["inputs"] = {}
        json_data["inputs"]["question"] = text
        json_data["inputs"]["image"] = img_base64
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json=json_data)
        return response.json()
    if task == "image-to-image":
        img_url = data["image"]
        img_data = requests.get(img_url, timeout=10).content
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data)
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "text-to-image":
        text = data["text"]
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json={"inputs": text})
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        name = str(uuid.uuid4())[:4]
        img.save(f"public/images/{name}.png")
        results = {}
        results["generated image"] = f"/images/{name}.png"
        return results
    if task == "image-segmentation":
        img_url = data["image"]
        img_data = requests.get(img_url, proxies=PROXY, timeout=10).content
        image = Image.open(BytesIO(img_data))
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data, proxies=PROXY)
        predicted = response.json()
        # generate different rgba colors for different classes
        colors = []
        for i in range(len(predicted)):
            colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 155))
        for i, pred in enumerate(predicted):
            label = pred["label"]
            mask = pred.pop("mask").encode("utf-8")
            # decode base64 to image and draw segmentation mask on image
            mask = base64.b64decode(mask)
            mask = Image.open(BytesIO(mask), mode='r')
            mask = mask.convert('L')

            layer = Image.new('RGBA', mask.size, colors[i])
            image.paste(layer, (0, 0), mask)
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        results = {}
        results["generated image with segmentation mask"] = f"/images/{name}.jpg"
        results["predicted"] = predicted
        return results
    if task == "object-detection":
        img_url = data["image"]
        img_data = requests.get(img_url, proxies=PROXY, timeout=10).content
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        r = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data, proxies=PROXY)
        predicted = r.json()
        image = Image.open(BytesIO(img_data))
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=2)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        response = {}
        response["generated image with predicted box"] = f"/images/{name}.jpg"
        response["predicted"] = predicted
        return response
    if task == "image-to-text":
        img_url = data["image"]
        img_data = requests.get(img_url, timeout=10).content
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data)
        results = {}
        if "generated_text" in response.json()[0]:
            results["generated text"] = response.json()[0].pop("generated_text")
        return results

    if task in ["image-classification"]:
        img_url = data["image"]
        img_data = requests.get(img_url, timeout=10).content
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=img_data)
        return response.json()


    # AUDIO tasks
    if task == "text-to-speech":
        text = data["text"]
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, json={"inputs": text})
        name = str(uuid.uuid4())[:4]
        with open(f"public/audios/{name}.flac", "wb") as f:
            f.write(response.content)
        return {"generated audio": f"/audios/{name}.flac"}
    if task in ["automatic-speech-recognition", "audio-to-audio", "audio-classification"]:
        audio_url = data["audio"]
        audio_type = audio_url.split(".")[-1]
        audio_data = requests.get(audio_url, timeout=10).content
        HUGGINGFACE_HEADERS["Content-Length"] = str(len(audio_data))
        HUGGINGFACE_HEADERS["Content-Type"] = f"audio/{audio_type}"
        response = requests.post(task_url, headers=HUGGINGFACE_HEADERS, data=audio_data)
        results = response.json()
        if task == "audio-to-audio":
            content = None
            type = None
            for k, v in results[0].items():
                if k == "blob":
                    content = base64.b64decode(v.encode("utf-8"))
                if k == "content-type":
                    type = "audio/flac".split("/")[-1]
            audio = AudioSegment.from_file(BytesIO(content))
            name = str(uuid.uuid4())[:4]
            audio.export(f"public/audios/{name}.{type}", format=type)
            return {"generated audio": f"/audios/{name}.{type}"}
        else:
            return results

def local_model_inference(model_id, data, task):
    task_url = f"{Model_Server}/models/{model_id}"
    
    # contronlet
    if model_id.startswith("lllyasviel/sd-controlnet-"):
        img_url = data["image"]
        text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if model_id.endswith("-control"):
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
        
    if task == "text-to-video":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated video"] = results.pop("path")
        return results

    # NLP tasks
    if task == "question-answering" or task == "sentence-similarity":
        response = requests.post(task_url, json=data)
        return response.json()
    if task in ["text-classification",  "token-classification", "text2text-generation", "summarization", "translation", "conversational", "text-generation"]:
        response = requests.post(task_url, json=data)
        return response.json()

    # CV tasks
    if task == "depth-estimation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated depth image"] = results.pop("path")
        return results
    if task == "image-segmentation":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        results["generated image with segmentation mask"] = results.pop("path")
        return results
    if task == "image-to-image":
        img_url = data["image"]
        response = requests.post(task_url, json={"img_url": img_url})
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "text-to-image":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated image"] = results.pop("path")
        return results
    if task == "object-detection":
        img_url = data["image"]
        img_data = requests.get(img_url, proxies=PROXY, timeout=10).content
        response = requests.post(task_url, json={"img_url": img_url})
        predicted = response.json()
        if "error" in predicted:
            return predicted
        image = Image.open(BytesIO(img_data))
        draw = ImageDraw.Draw(image)
        labels = list(item['label'] for item in predicted)
        color_map = {}
        for label in labels:
            if label not in color_map:
                color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
        for label in predicted:
            box = label["box"]
            draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]], width=2)
            draw.text((box["xmin"]+5, box["ymin"]-15), label["label"], fill=color_map[label["label"]])
        name = str(uuid.uuid4())[:4]
        image.save(f"public/images/{name}.jpg")
        results = {}
        results["generated image with predicted box"] = f"/images/{name}.jpg"
        results["predicted"] = predicted
        return results
    if task in ["image-classification", "image-to-text", "document-question-answering", "visual-question-answering"]:
        img_url = data["image"]
        text = None
        if "text" in data:
            text = data["text"]
        response = requests.post(task_url, json={"img_url": img_url, "text": text})
        results = response.json()
        return results
    # AUDIO tasks
    if task == "text-to-speech":
        response = requests.post(task_url, json=data)
        results = response.json()
        if "path" in results:
            results["generated audio"] = results.pop("path")
        return results
    if task in ["automatic-speech-recognition", "audio-to-audio", "audio-classification"]:
        audio_url = data["audio"]
        response = requests.post(task_url, json={"audio_url": audio_url})
        return response.json()


def model_inference(model_id, data, task):
    use_huggingface_service = False
    use_local_service = False
    huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
    r = requests.get(huggingfaceStatusUrl, headers=HUGGINGFACE_HEADERS, proxies=PROXY)
    logger.info("Huggingface Status: " + str(r.json()))
    if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
        use_huggingface_service = True
    
    localStatusUrl = f"{Model_Server}/status/{model_id}"
    r = requests.get(localStatusUrl)
    logger.info("Local Server Status: " + str(r.json()))
    if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
        use_local_service = True
    
    try:
        if use_local_service:
            inference_result = local_model_inference(model_id, data, task)
        elif use_huggingface_service:
            inference_result = hugginhface_model_inference(model_id, data, task)
        else:
            inference_result = {"error":{"message": "there are no services available locally or at Huggingface."}}
    except Exception as e:
        traceback.print_exc()
        inference_result = {"error":{"message": str(e)}}
    return inference_result

def get_id_reason(choose_str):
    reason = field_extract(choose_str, "reason")
    id = field_extract(choose_str, "id")
    choose = {"id": id, "reason": reason}
    return id.strip(), reason.strip(), choose


def get_model_id_status(model_id, url, headers, queue):
    if "huggingface" in url:
        r = requests.get(url, headers=headers, proxies=PROXY)
    else:
        r = requests.get(url)
    if r.status_code == 200 and "loaded" in r.json() and r.json()["loaded"]:
        queue.put((model_id, True))
    else:
        queue.put((model_id, False))

def get_avaliable_model_ids(candidates, topk=5):
    all_available_model_ids = []
    processes = []
    result_queue = multiprocessing.Queue()

    for candidate in candidates:
        model_id = candidate["id"]
        huggingfaceStatusUrl = f"https://api-inference.huggingface.co/status/{model_id}"
        process = multiprocessing.Process(target=get_model_id_status, args=(model_id, huggingfaceStatusUrl, HUGGINGFACE_HEADERS, result_queue))
        processes.append(process)
        process.start()

        localStatusUrl = f"{Model_Server}/status/{model_id}"
        process = multiprocessing.Process(target=get_model_id_status, args=(model_id, localStatusUrl, {}, result_queue))
        processes.append(process)
        process.start()
        
    result_count = len(processes)
    while result_count:
        model_id, status = result_queue.get()
        if status and model_id not in all_available_model_ids:
            all_available_model_ids.append(model_id)
        if len(all_available_model_ids) >= topk:
            break
        result_count -= 1

    for process in processes:
        process.join()

    return all_available_model_ids

def colloct_result(command, choose, inference_result):
    result = {"task": command}
    result["inference result"] = inference_result
    result["choose model result"] = choose
    logger.info(f"inference result: {inference_result}")
    return result


def run_task(input, command, results):
    id = command["id"]
    args = command["args"]
    task = command["task"]
    deps = command["dep"]
    if deps[0] != -1:
        dep_tasks = [results[dep] for dep in deps]
    else:
        dep_tasks = []
    
    logger.info(f"Run task: {id} - {task}")
    logger.info("Deps: " + json.dumps(dep_tasks))

    if deps[0] != -1:
        if "image" in args and "<GENERATED>-" in args["image"]:
            resource_id = int(args["image"].split("-")[1])
            if "generated image" in results[resource_id]["inference result"]:
                args["image"] = results[resource_id]["inference result"]["generated image"]
        if "audio" in args and "<GENERATED>-" in args["audio"]:
            resource_id = int(args["audio"].split("-")[1])
            if "generated audio" in results[resource_id]["inference result"]:
                args["audio"] = results[resource_id]["inference result"]["generated audio"]
        if "text" in args and "<GENERATED>-" in args["text"]:
            resource_id = int(args["text"].split("-")[1])
            if "generated text" in results[resource_id]["inference result"]:
                args["text"] = results[resource_id]["inference result"]["generated text"]

    text = image = audio = None
    for dep_task in dep_tasks:
        if "generated text" in dep_task["inference result"]:
            text = dep_task["inference result"]["generated text"]
            logger.info("Detect the generated text of dependency task (from results):" + text)
        elif "text" in dep_task["task"]["args"]:
            text = dep_task["task"]["args"]["text"]
            logger.info("Detect the text of dependency task (from args): " + text)
        if "generated image" in dep_task["inference result"]:
            image = HTTP_Server + dep_task["inference result"]["generated image"]
            logger.info("Detect the generated image of dependency task (from results): " + image)
        elif "image" in dep_task["task"]["args"]:
            image = dep_task["task"]["args"]["image"]
            logger.info("Detect the image of dependency task (from args): " + image)
        if "generated audio" in dep_task["inference result"]:
            audio = HTTP_Server + dep_task["inference result"]["generated audio"]
            logger.info("Detect the generated audio of dependency task (from results): " + audio)
        elif "audio" in dep_task["task"]["args"]:
            audio = dep_task["task"]["args"]["audio"]
            logger.info("Detect the audio of dependency task (from args): " + audio)


    if "image" in args and "<GENERATED>" in args["image"]:
        if image:
            args["image"] = image
    if "audio" in args and "<GENERATED>" in args["audio"]:
        if audio:
            args["audio"] = audio
    if "text" in args and "<GENERATED>" in args["text"]:
        if text:
            args["text"] = text

    for resource in ["image", "audio"]:
        if resource in args and not args[resource].startswith("http") and len(args[resource]) > 0:
            args[resource] = f"{HTTP_Server}{args[resource]}"
    
    if "-text-to-image" in command['task'] and "text" not in args:
        logger.info("control-text-to-image task, but text is empty, so we use control-generation instead.")
        control = task.split("-")[0]
        
        if control == "seg":
            task = "image-segmentation"
            command['task'] = task
        elif control == "depth":
            task = "depth-estimation"
            command['task'] = task
        else:
            task = f"{control}-control"

    command["args"] = args
    logger.info(f"parsed task: {command}")

    if task.endswith("-text-to-image"):
        control = task.split("-")[0]
        best_model_id = f"lllyasviel/sd-controlnet-{control}"
        reason = "ControlNet is the best model for this task."
        choose = {"id": best_model_id, "reason": reason}
        logger.info(f"chosen model: {choose}")
    elif task.endswith("-control"):
        best_model_id = task
        reason = "ControlNet tools"
        choose = {"id": best_model_id, "reason": reason}
        logger.info(f"chosen model: {choose}")
    else:
        if task not in MODELS_MAP:
            record_case(success=False, **{"input": input, "task": command, "reason": f"task not support: {command['task']}", "op":"message"})
            inference_result = {"error": f"{command['task']} not found in available tasks."}
            results[id] = colloct_result(command, choose, inference_result)
            return False

        candidates = MODELS_MAP[task][:40]
        all_avaliable_model_ids = get_avaliable_model_ids(candidates)
        logger.info(f"avaliable models on {command['task']}: {all_avaliable_model_ids}")

        # controlnet direct choose
        if len(all_avaliable_model_ids) == 0:
            record_case(success=False, **{"input": input, "task": command, "reason": f"no available models: {command['task']}", "op":"message"})
            inference_result = {"error": f"no available models on {command['task']} task."}
            results[id] = colloct_result(command, "", inference_result)
            return False

            
        if len(all_avaliable_model_ids) == 1:
            best_model_id = all_avaliable_model_ids[0]
            reason = "Only one model available."
            choose = {"id": best_model_id, "reason": reason}
            logger.info(f"chosen model: {choose}")
        else:
            cand_models_info = []
            for model in candidates:
                model_info = {}
                if model["id"] not in all_avaliable_model_ids:
                    continue
                if "likes" in model:
                    model_info["likes"] = model["likes"]
                model_info["id"] = model["id"]
                if "description" in model:
                    model_info["description"] = model["description"][:100]
                if "language" in model:
                    model_info["language"] = model["language"]
                if "tags" in model:
                    model_info["tags"] = model["tags"]
                cand_models_info.append(model_info)

            choose_str = choose_model(input, command, cand_models_info)
            logger.info(f"chosen model: {choose_str}")
            try:
                choose_str = find_json(choose_str)
                best_model_id, reason, choose  = get_id_reason(choose_str)
            except Exception as e:
                print(e)
                choose = json.loads(choose)
                reason = choose["reason"]
                best_model_id = choose["id"]
    inference_result = model_inference(best_model_id, args, command['task'])

    if "error" in inference_result:
        record_case(success=False, **{"input": input, "task": command, "reason": f"inference error: {inference_result['error']}", "op":"message"})
        results[id] = colloct_result(command, choose, inference_result)
        return False
    
    results[id] = colloct_result(command, choose, inference_result)
    return True

def has_dep(command):
    args = command["args"]
    for k, v in args.items():
        if "<GENERATED>" in v:
            return True
    return False

def chat_huggingface(messages):
    start = time.time()
    context = messages[1:-1]
    input = messages[-1]["content"]

    task_str = parse_task(context, input).strip()

    if task_str == "[]":
        record_case(success=False, **{"input": input, "task": [], "reason": "task parsing fail: empty", "op": "chitchat"})
        response = chitchat(messages)
        return {"message": response}
    try:
        logger.info(task_str)
        tasks = json.loads(task_str)
    except Exception as e:
        logger.info(e)
        response = chitchat(messages)
        record_case(success=False, **{"input": input, "task": task_str, "reason": "task parsing fail", "op":"chitchat"})
        return {"message": response}

    results = {}
    processes = []
    tasks = tasks[:]
    with  multiprocessing.Manager() as manager:
        d = manager.dict()
        retry = 0
        while True:
            num_process = len(processes)
            for task in tasks:
                if not has_dep(task):
                    task["dep"] = [-1]
                dep = task["dep"]
                if len(list(set(dep).intersection(d.keys()))) == len(dep) or dep[0] == -1:
                    tasks.remove(task)
                    process = multiprocessing.Process(target=run_task, args=(input, task, d))
                    process.start()
                    processes.append(process)
            if num_process == len(processes):
                time.sleep(0.5)
                retry += 1
            if retry > 160:
                logger.info("User has waited too long, Loop break.")
                break
            if len(tasks) == 0:
                break
        for process in processes:
            process.join()
        
        results = d.copy()
    logger.info(results)
    response = response_results(input, results).strip()

    end = time.time()
    during = end - start

    logger.info(f"response to user: {response}")
    answer = {"message": response}
    record_case(success=True, **{"input": input, "task": task_str, "results": results, "response": response, "during": during, "op":"response"})
    return answer

if __name__ == "__main__":
    inputs = [
        # "Given a collection of image A: /examples/cat.jpg, B: /examples/three-zebra.jpg, C: /examples/zebra.jpg, please tell me how many zebras in these picture?"
        # "Can you give me a picture of a small bird flying in the sky with trees and clouds. Generate a high definition image if possible.",
        # "Please answer all the named entities in the sentence: Iron Man is a superhero appearing in American comic books published by Marvel Comics. The character was co-created by writer and editor Stan Lee, developed by scripter Larry Lieber, and designed by artists Don Heck and Jack Kirby.",
        # "please dub for me: 'Iron Man is a superhero appearing in American comic books published by Marvel Comics. The character was co-created by writer and editor Stan Lee, developed by scripter Larry Lieber, and designed by artists Don Heck and Jack Kirby.'"
        "Given an image: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg, please answer the question: What is on top of the building?",
        "What's in the picture?  https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg"
        ]
    for input in inputs:
        logger.info(chat_huggingface(input))                                    