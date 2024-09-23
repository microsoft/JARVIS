
import os
import json
import click
import asyncio
import aiohttp
import logging
import emoji

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

class RateLimitError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ContentFormatError(Exception):
    def __init__(self, message):
        super().__init__(message)

@click.command()
@click.option("--data_dir", default="data_huggingface", help="The directory of the data.")
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.1)
@click.option("--api_addr", type=str, default="localhost")
@click.option("--api_port", type=int, default=4000)
@click.option("--api_key", type=str, default="your api key")
@click.option("--multiworker", type=int, default=1)
@click.option("--llm", type=str, default="gpt-4")
@click.option("--use_demos", type=int, default=0)
@click.option("--reformat", type=bool, default=False)
@click.option("--reformat_by", type=str, default="self")
@click.option("--tag", type=bool, default=False)
@click.option("--dependency_type", type=str, default="resource")
@click.option("--log_first_detail", type=bool, default=False)
def main(data_dir, temperature, top_p, api_addr, api_key, api_port, multiworker, llm, use_demos, reformat, reformat_by, tag, dependency_type, log_first_detail):
    assert dependency_type in ["resource", "temporal"], "Dependency type not supported"
    if dependency_type == "resource":
        assert data_dir != "data_dailylifeapis", "Resource dependency type only support data_huggingface and data_multimedia"

    arguments = locals()
    url = f"http://{api_addr}:{api_port}/v1/chat/completions"
    header = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prediction_dir = f"{data_dir}/predictions{f'_use_demos_{use_demos}' if use_demos and tag else ''}{f'_reformat_by_{ reformat_by}' if reformat and tag else ''}"
    wf_name = f"{prediction_dir}/{llm}.json"
    
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    has_inferenced = []
    if os.path.exists(wf_name):
        rf = open(wf_name, "r")
        for line in rf:
            data = json.loads(line)
            has_inferenced.append(data["id"])
        rf.close()

    rf_ur = open(f"{data_dir}/user_requests.json", "r")
    inputs = []
    for line in rf_ur:
        input = json.loads(line)
        if input["id"] not in has_inferenced:
            inputs.append(input)
    rf_ur.close()

    wf = open(wf_name, "a")
    
    tool_list = json.load(open(f"{data_dir}/tool_desc.json", "r"))["nodes"]
    if "input-type" not in tool_list[0]:
        assert dependency_type == "temporal", "Tool type is not ignored, but the tool list does not contain input-type and output-type"
    if dependency_type == "temporal":
        for tool in tool_list:
            parameter_list = []
            for parameter in tool["parameters"]:
                parameter_list.append(parameter["name"])
            tool["parameters"] = parameter_list

    # log llm name in format
    formatter = logging.Formatter(f"%(asctime)s - [ {llm} ] - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(f"{prediction_dir}/{llm}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # logging all args
    logger.info(f"Arguments: {arguments}")

    demos = []
    if use_demos:
        if dependency_type == "temporal":
            demos_id = [ "38563456", "27267145", "91005535"]
        else:
            if "huggingface" in data_dir: 
                demos_id = [ "10523150", "14611002", "22067492"]
            elif "multimedia" in data_dir:
                demos_id = [ "30934207", "20566230", "19003517"]
        demos_id = demos_id[:use_demos]
        logger.info(f"Use {len(demos_id)} demos: {demos_id}")
        demos_rf = open(f"{data_dir}/data.json", "r")
        for line in demos_rf:
            data = json.loads(line)
            if data["id"] in demos_id:
                if dependency_type == "temporal":
                    demo = {
                        "user_request": data["user_request"],
                        "result":{
                            "task_steps": data["task_steps"],
                            "task_nodes": data["task_nodes"],
                            "task_links": data["task_links"]
                        }
                    }
                else:
                    demo = {
                        "user_request": data["user_request"],
                        "result":{
                            "task_steps": data["task_steps"],
                            "task_nodes": data["task_nodes"]
                        }
                    }
                demos.append(demo)
        demos_rf.close()

    tool_string = "# TASK LIST #:\n"
    for k, tool in enumerate(tool_list):
        tool_string += json.dumps(tool) + "\n"
    
    sem = asyncio.Semaphore(multiworker)

    async def inference_wrapper(input, url, header, temperature, top_p, tool_string, wf, llm, demos, reformat, reformat_by, dependency_type, log_detail = False):
        async with sem:
            await inference(input, url, header, temperature, top_p, tool_string, wf, llm, demos, reformat, reformat_by, dependency_type, log_detail)

    if len(inputs) == 0:
        logger.info("All Completed!")
        return
    else:
        logger.info(f"Detected {len(has_inferenced)} has been inferenced,")
        logger.info(f"Start inferencing {len(inputs)} tasks...")
    
    loop = asyncio.get_event_loop()

    if log_first_detail:
        tasks = [inference_wrapper(inputs[0], url, header, temperature, top_p, tool_string, wf, llm, demos, reformat, reformat_by, dependency_type, log_detail=True)]
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        inputs = inputs[1:]

    tasks = []
    for input in inputs:
        tasks.append(inference_wrapper(input, url, header, temperature, top_p, tool_string, wf, llm, demos, reformat, reformat_by, dependency_type))

    results += loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    failed = []
    done = []
    for result in results:
        if isinstance(result, Exception):
            failed.append(result)
        else:
            done.append(result)
    logger.info(f"Completed: {len(done)}")
    logger.info(f"Failed: {len(failed)}")
    loop.close()

async def inference(input, url, header, temperature, top_p, tool_string, wf, llm, demos, reformat, reformat_by, dependency_type, log_detail = False):
    user_request = input["user_request"]
    if dependency_type == "resource":
        prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]} """
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
    else:
        prompt = """\n# GOAL #:\nBased on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;"""

    if len(demos) > 0:
        prompt += "\n"
        for demo in demos:
            prompt += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""
    
    prompt += """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""

    final_prompt = tool_string + prompt.replace("{{user_request}}", user_request)
    payload = json.dumps({
        "model": f"{llm}",
        "messages": [
            {
            "role": "user",
            "content":  final_prompt
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 2000,
        "stream": False,
        "stop": None
    })
    try:
        result = await get_response(url, header, payload, input['id'], reformat, reformat_by, dependency_type, log_detail)
    except Exception as e:
        logger.info(f"Failed #id {input['id']}: {type(e)} {e}")
        raise e
    logger.info(f"Success #id {input['id']}")
    input["result"] = result
    wf.write(json.dumps(input) + "\n")
    wf.flush()

async def get_response(url, header, payload, id, reformat, reformat_by, dependency_type, log_detail=False):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=header, data=payload, timeout=300) as response:
            resp = await response.json()

    if response.status == 429:
        raise RateLimitError(f"{resp}")
    if response.status != 200:
        raise Exception(f"{resp}")
    
    if log_detail:
        logger.info(json.loads(payload)["messages"][0]["content"])
        logger.info(resp["choices"][0]["message"]["content"])

    oring_content = resp["choices"][0]["message"]["content"]
    oring_content = oring_content.replace("\n", "")
    oring_content = oring_content.replace("\_", "_")
    content = oring_content.replace("\\", "")

    start_pos = content.find("RESULT #:")
    if start_pos!=-1:
        content = content[start_pos+len("RESULT #:"):]
        
    content = content[content.find("{"):content.rfind("}")+1]
    try:
        content = json.loads(content)
        if isinstance(content, list) and len(content):
            merge_content = {}
            for c in content:
                for k, v in c.items():
                    merge_content[k].extend(v) if k in merge_content else merge_content.update({k: v})
        return content
    except json.JSONDecodeError as e:
        if reformat:
            if dependency_type == "resource":
                prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps and task nodes;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. You must output the result in this schema: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
            else:
                prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
            prompt = prompt.replace("{{illegal_result}}", oring_content)
            payload = json.loads(payload)
            if reformat_by != "self":
                payload["model"] = reformat_by

            if log_detail:
                logger.info(f"{emoji.emojize(':warning:')}  #id {id} Illegal JSON format: {content}")
                logger.info(f"{emoji.emojize(':sparkles:')} #id {id} Detected illegal JSON format, try to reformat by {payload['model']}...")

            payload["messages"][0]["content"] = prompt
            payload = json.dumps(payload)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=header, data=payload, timeout=120) as response:
                    resp = await response.json()

            if response.status == 429:
                raise RateLimitError(f"{resp}")
            if response.status != 200:
                raise Exception(f"{resp}")
            
            if log_detail:
                logger.info(json.loads(payload)["messages"][0]["content"])
                logger.info(resp["choices"][0]["message"]["content"])

            content = resp["choices"][0]["message"]["content"]
            content = content.replace("\n", "")
            content = content.replace("\_", "_")
            start_pos = content.find("STRICT JSON FORMAT #:")
            if start_pos!=-1:
                content = content[start_pos+len("STRICT JSON FORMAT #:"):]

            content = content[content.find("{"):content.rfind("}")+1]
            try:
                content = json.loads(content)
                return content
            except json.JSONDecodeError as e:
                raise ContentFormatError(f"{content}")
        else:
            raise ContentFormatError(f"{content}")

if __name__ == "__main__":
    main()