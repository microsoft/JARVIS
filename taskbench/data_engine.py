import os
import random
import traceback
import uuid
import json
import networkx as nx
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import warnings

from graph_sampler import GraphSampler
import click
import matplotlib.pyplot as plt
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

@click.command()
@click.option("--temperature", type=float, default=0.7)
@click.option("--top_p", type=float, default=1)
@click.option("--check", type=bool, default=False)
@click.option("--data_dir", type=str, default=None)
@click.option("--graph_desc", type=str, default=None)
@click.option("--tool_desc", type=str, default=None)
@click.option("--api_addr", type=str, default="localhost")
@click.option("--api_port", type=int, default=4000)
@click.option("--api_key", type=str, default="your api key")
@click.option("--play", type=bool, default=False)
@click.option("--method", type=str, default=None)
@click.option("--tool_number", type=int, default=None)
@click.option("--number_of_samples", type=int, default=5000)
@click.option("--seed", type=int, default=-1)
@click.option("--save_figure", type=bool, default=False)
@click.option("--multiworker", type=int, default=1)
@click.option("--llm", type=str, default="gpt-4")
@click.option("--use_async", type=bool, default=False)
@click.option("--dependency_type", type=str, default="resource")
def main(temperature, top_p, check, graph_desc, tool_desc, api_addr, api_port, api_key, play, method, tool_number, number_of_samples, seed, data_dir, save_figure, multiworker, llm, use_async, dependency_type):
    args = locals()
    url = f"http://{api_addr}:{api_port}/v1/chat/completions"
    header = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    now = datetime.now()

    if data_dir:
        if os.path.exists(data_dir):
            if graph_desc or tool_desc:
                warnings.warn(f"Data directory {data_dir} already exists, tool graph and tool desc will be ignored.")
            graph_desc = f"{data_dir}/graph_desc.json"
            tool_desc = f"{data_dir}/tool_desc.json"
    else:
        data_dir = f"result_{now.strftime('%Y%m%d%H%M%S')}_{llm}_t{temperature}_p{top_p}{'_check' if check else ''}".replace(".", "_")
    
    assert data_dir and graph_desc and tool_desc

    if seed > -1:
        random.seed(seed)
    
    tool_list = json.load(open(tool_desc, "r"))["nodes"]
    tools = {}
    if dependency_type == "resource":
        assert "input-type" in tool_list[0] and "output-type" in tool_list[0], "Input and output types are not defined"
        for tool in tool_list:
            tools[tool["id"]] = {"id": tool["id"], "desc": tool["desc"], "input-type": tool["input-type"], "output-type": tool["output-type"]}
    elif dependency_type == "temporal":
        assert "parameters" in tool_list[0], "Parameters are not defined"
        for tool in tool_list:
            tools[tool["id"]] = {"id": tool["id"], "desc": tool["desc"], "parameters": tool["parameters"]}
    else:
        raise ValueError(f"Unsupported dependency type: {dependency_type}")

    sampler = GraphSampler(file_name=graph_desc)

    if play:
        assert method is not None
        assert tool_number is not None
        result = asyncio.run(sample(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, "./", None, dependency_type))
        logger.info(json.dumps(result, indent=2))
        return

    figure_dir = None

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        shutil.copy(graph_desc, f"{data_dir}/graph_desc.json")
        shutil.copy(tool_desc, f"{data_dir}/tool_desc.json")

    output = f"{data_dir}/data_raw.json"
    statistics_output = f"{data_dir}/statistics.json"

    file_handler = logging.FileHandler(f"{data_dir}/data_engine.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(json.dumps(args))

    if save_figure:
        figure_dir = f"{data_dir}/task_graphs"
        os.makedirs(figure_dir, exist_ok=True)

    wf = open(output, "a")
    statistics_wf = open(f"{statistics_output}", "a")
    args["start_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
    statistics_wf.write(json.dumps(args) + "\n")

    method_weights = {
        "single": 3,
        "chain": 7,
        "dag": 8,
    }

    number_weights = {
        2: 0.1,
        3: 0.2,
        4: 0.3,
        5: 0.2,
        6: 0.1,
        7: 0.05,
        8: 0.025,
        9: 0.025,
        10: 0.025,
    }

    statistics = {"total": 0, "avg_time_per_sample": 0, "success": 0, "fail": 0}
    done, failed = [], []
    if use_async:
        # coroutine with Semaphore
        sem = asyncio.Semaphore(multiworker)
        async def sample_with_statistics(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, statistics, now, dependency_type):
            async with sem:  # semaphore limits num of simultaneous sampling
                if statistics["total"] % 100 == 0 and statistics["total"] != 0:
                    logger.info(json.dumps(statistics, indent=2))
                    statistics_wf.write(json.dumps(statistics) + "\n")
                try:
                    await sample(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, dependency_type)
                except Exception as e:
                    statistics["total"] += 1
                    statistics["fail"] += 1
                    if str(type(e)) not in statistics:
                        statistics[str(type(e))] = 0
                    statistics[str(type(e))] += 1
                    raise e
                statistics["total"] += 1
                statistics["success"] += 1
                statistics["avg_time_per_sample"] = str((datetime.now() - now) / statistics["success"])

        async def run(url, header, llm, temperature, top_p, check, sampler, tools, figure_dir, wf, statistics, now, dependency_type):
            method = random.choices(list(method_weights.keys()), weights=list(method_weights.values()))[0]
            if method == "single":
                tool_number = 1
            else:
                tool_number = random.choices(list(number_weights.keys()), weights=list(number_weights.values()))[0]
                if method == "dag":
                    tool_number = max(tool_number, 3)
            await sample_with_statistics(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, statistics, now, dependency_type)

        tasks = []
        for _ in range(number_of_samples):
            tasks.append(run(url, header, llm, temperature, top_p, check, sampler, tools, figure_dir, wf, statistics, now, dependency_type))

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

        for result in results:
            if isinstance(result, Exception):
                failed.append(result)
            else:
                done.append(result)
    else:
        # multi-thread with ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=multiworker)
        def sample_with_statistics(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, statistics, now, dependency_type):
            if statistics["total"] % 100 == 0 and statistics["total"] != 0:
                logger.info(json.dumps(statistics, indent=2))
                statistics_wf.write(json.dumps(statistics) + "\n")
            try:
                asyncio.run(sample(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, dependency_type))
            except Exception as e:
                statistics["total"] += 1
                statistics["fail"] += 1
                if str(type(e)) not in statistics:
                    statistics[str(type(e))] = 0
                statistics[str(type(e))] += 1
                raise e
            statistics["total"] += 1
            statistics["success"] += 1
            statistics["avg_time_per_sample"] = str((datetime.now() - now) / statistics["success"])

        def run(url, header, llm, temperature, top_p, check, sampler, tools, figure_dir, wf, statistics, now, dependency_type):
            method = random.choices(list(method_weights.keys()), weights=list(method_weights.values()))[0]
            if method == "single":
                tool_number = 1
            else:
                tool_number = random.choices(list(number_weights.keys()), weights=list(number_weights.values()))[0]
                if method == "dag":
                    tool_number = max(tool_number, 3)
            sample_with_statistics(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, statistics, now, dependency_type)

        tasks = []
        for _ in range(number_of_samples):
            tasks.append(executor.submit(run, url, header, llm, temperature, top_p, check, sampler, tools, figure_dir, wf, statistics, now, dependency_type))
        for future in as_completed(tasks):
            try:
                future.result()
                done.append(future)
            except Exception as e:
                failed.append(future)

    statistics_wf.write(json.dumps(statistics) + "\n")
    logger.info(f"Done: {len(done)}, Failed: {len(failed)}")

class RateLimitError(Exception):
    def __init__(self, message):
        super().__init__(message)

class ContentFormatError(Exception):
    def __init__(self, message):
        super().__init__(message)

async def sample(url, header, llm, temperature, top_p, check, tool_number, sampler, tools, method, figure_dir, wf, dependency_type):
    start_time = datetime.now()
    sample_id = str(uuid.uuid4().int)[:8]
    sub_G = sampler.sample_subgraph(tool_number, sample_method=method)

    tool_list = list(sub_G.nodes)
    tool_edge = list(sub_G.edges)  
    seed = random.randint(0, 1000000)
    sampled_tools_string = "Given a tool graph with tools as nodes, and invoking chains between tools as edges. The following tools (nodes) are available with their corresponding descriptions and input/outputs types:\n"
    for k, tool in enumerate(tool_list):
        sampled_tools_string += f"Node {k+1}:" + json.dumps(tools[tool]) + "\n"

    sampled_links_string = "These tools can be connected as follows (the directed edges are invoking chains among tools):\n"
    for k, edge in enumerate(tool_edge):
        sampled_links_string += f"Edge: " + edge[0] + " -> " + edge[1] + "\n"
    prompt = """\nBased on the above tool graph, please be skillful to generate the according task steps, user request and tool invoking graph. \nRequirements: \n1. the generated user request should be somewhat clear, self-contained (user-specified text, image, video, audio, content should be contained in the request) and practical (help users solve a practical problem); \n2. the task steps must be strictly aligned with the tool graph (nodes and edges) and reasonable, the tool invoking graph must align with task steps, also with the given tool graph; \n3. the user request just can be decomposed into task steps solved by the tool invoking graph; \n4. each task step corresponds to a tool node in the tool graph and tool invoking graph, and the number of task steps must be same with the nodes. Each tool node can only be used once; \n5. if need image/audio/video resources in user request, please use files 'example.[jpg/mp4/wav/png]'; \n6. the dependencies among task steps must align with the edges of tool graph and tool invoking graph; \n7. the number and types of tool parameters in the generated tool invoking graph need to be consistent with the pre-defined input/outputs types of the tools. \nNow please generate your result (with random seed {""" + f"{seed}"+ """}) in a compact JSON format"""
    if dependency_type == "resource":
        prompt += """{"task_steps": [ step description of one or more steps ], "user_request": "your high-quality and self-contained synthesized request", "invoking_graph": {"nodes": [{"id": "tool name", "input": [ either user-specified text or resource file 'example.[jpg/mp4/wav/png' ] in the above user request, or the dependent tool name whose output is required by this node ]}], "links": [{"source": "tool name i", "target": "tool name j"}]}"""
    else:
        prompt += """{"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "user_request": "your high-quality, concrete and self-contained synthesized request, with explicit parameter values", "invoking_graph": {"nodes": [{"id": "tool name", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "links": [{"source": "tool name i", "target": "tool name j"}]}"""
    if check:
        prompt += """, "check_by_teacher": "This field is filled by your strict and well-trained teacher, minor mistakes are complete intolerable to him. He evaluated whether your synthesized user request, tool invoking graph are valid and whether they are aligned with the given tool graph (strictly checked step by step according to the above requirements). Some comments from him place here (start with 'Let me check your result step by step, and evaluate the 'Executable' and 'Correct' of the tool invoking graph (Executable means that the tool invoking graph executed successfully, regardless of alignment with the given tool graph. While Correct implies that the tool invoking graph are not only 'Executable' but also strictly consistent (with strictly same nodes and same edges) with the given tool graph). After carefully evaluating, found some mistakes:' and end with a conclusion: 'Conclusion: Executable: no/yes, Correct: no/yes'.)"""
    prompt += "}:"

    final_prompt = sampled_tools_string + sampled_links_string + prompt

    if dependency_type == "temporal":
        final_prompt = final_prompt.replace("tool", "API")

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
        "presence_penalty": 0,
        "max_tokens": 2500,
        "stream": False,
        "stop": None
    })
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=header, data=payload, timeout=120) as response:
                resp = await response.json()

        if response.status == 429:
            raise RateLimitError(f"{resp}")
        if response.status != 200:
            raise Exception(f"{resp}")

        content = resp["choices"][0]["message"]["content"]
        content = content.replace("\n", "")
        json_start = 0
        json_end = len(content)
        
        if "```json" in content:
            json_start = content.find("```json") + 7
        if "```" in content:
            json_end = content.rfind("```")
        content = content[json_start:json_end]
        logger.info(content)
        try:
            content = json.loads(content)
        except json.JSONDecodeError as e:
            raise ContentFormatError(f"{content}")

        if dependency_type == "resource":
            sampled_nodes = [{"id": tool, "input-type": tools[tool]["input-type"], "output-type": tools[tool]["output-type"]} for tool in tool_list]
        else:
            sampled_nodes = [{"id": tool, "parameters": tools[tool]["parameters"]} for tool in tool_list]

        sampled_links = [{"source": edge[0], "target": edge[1]} for edge in tool_edge]
        sampled_nodes = sorted(sampled_nodes, key=lambda x: x["id"])
        sampled_links = sorted(sampled_links, key=lambda x: (x["source"], x["target"]))

        content["invoking_graph"]["nodes"] = sorted(content["invoking_graph"]["nodes"], key=lambda x: x["id"])
        content["invoking_graph"]["links"] = sorted(content["invoking_graph"]["links"], key=lambda x: (x["source"], x["target"]))
            
        result = {"id": sample_id, "seed": seed, "method": method, "number_of_tools": tool_number, "sampled_nodes": sampled_nodes, "sampled_links": sampled_links, "result": content}
        
        if wf:
            wf.write(json.dumps(result) + "\n")

        if figure_dir:
            plt.figure()
            pos = nx.circular_layout(sub_G)
            nx.draw_networkx_nodes(sub_G, pos, node_color="skyblue", node_size=300)
            nx.draw_networkx_edges(sub_G, pos, arrows=True)
            nx.draw_networkx_labels(sub_G, pos, font_size=8)
            plt.tight_layout()
            plt.savefig(f"{figure_dir}/{sample_id}.jpg")
            plt.close()

        sampled_nodes_ids = [node["id"] for node in sampled_nodes]
        generated_nodes_ids = [node["id"] for node in content["invoking_graph"]["nodes"]]

        end_time = datetime.now()
        logger.info(f"Sample {sample_id} finished, time cost: {end_time - start_time}")
        if sampled_links == content["invoking_graph"]["links"] and sampled_nodes_ids == generated_nodes_ids:
            logger.info("Check success: invoking graph and sampled graph are aligned.")
        elif sampled_nodes_ids != generated_nodes_ids:
            logger.info("Check fail: mismatched nodes")
            logger.info("Sampled node:\n" + json.dumps(sampled_nodes_ids, indent=2))
            logger.info("Generated node:\n" + json.dumps(generated_nodes_ids, indent=2))
            logger.info(f"Sample {sample_id}:\n{json.dumps(result, indent=2)}")
        else:
            logger.info("Check fail: mismatched links")
            logger.info("Sampled link:\n" + json.dumps(sampled_links, indent=2))
            logger.info("Generated link:\n" + json.dumps(content["invoking_graph"]["links"], indent=2))
            logger.info(f"Sample {sample_id}:\n{json.dumps(result, indent=2)}")
    except Exception as e:
        logger.info(f"Failed: {type(e)}")
        print(traceback.format_exc())
        raise e
    return result

if __name__ == "__main__":
    main()