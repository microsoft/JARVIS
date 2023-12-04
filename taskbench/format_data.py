import json
import click
import traceback

def formulate_sample(data, dependency_type):
    try:
        user_request = data["result"]["user_request"]
        invoking_graph = data["result"]["invoking_graph"]
        task_steps = data["result"]["task_steps"]
        nodes = invoking_graph["nodes"]
        links = invoking_graph["links"]
        user_request = data["result"]["user_request"]
        if "check_by_teacher" in data["result"]:
            check_by_teacher = data["result"]["check_by_teacher"]
        else:
            check_by_teacher = invoking_graph["check_by_teacher"]
        for node in nodes:
            node["task"] = node["id"]
            node.pop("id")
            if dependency_type == "resource":
                node["task"] = node["task"].replace("_", " ")
                node["arguments"] = node["input"]
                node.pop("input")

        for node in nodes:
            assert isinstance(node, dict)
            assert "task" in node
            assert "arguments" in node
            if isinstance(node["arguments"], str) and node["arguments"].startswith("<node-"):
                node["arguments"] = [node["arguments"]]
            assert isinstance(node["arguments"], list), node["arguments"]
            if dependency_type == "resource":
                assert len(node["arguments"]) <= 2
                for i, argument in enumerate(node["arguments"]):
                    for j, n in enumerate(nodes):
                        if n["task"] in argument:
                            node["arguments"][i] = f"<node-{j}>"
                            break
        for link in links:
            assert isinstance(link, dict)
            assert "source" in link
            assert "target" in link
            if dependency_type == "resource":
                link["source"] = link["source"].replace("_", " ")
                link["target"] = link["target"].replace("_", " ")
        assert isinstance(task_steps, list)
        assert isinstance(nodes, list)
        assert len(nodes) == len(task_steps)
        assert isinstance(user_request, str)
        assert isinstance(check_by_teacher, str)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None, None, None, None, None
    return user_request, task_steps, links, nodes, check_by_teacher

@click.command()
@click.option('--data_dir', default='data_huggingface', help='Path to the data directory')
@click.option('--dependency_type', default='resource')
def formulate(data_dir, dependency_type):
    rf = open(f"{data_dir}/data_raw.json", "r")
    wf_format = open(f"{data_dir}/data.json", "w")
    wf_error = open(f"{data_dir}/data_error.json", "w")
    wf_ur = open(f"{data_dir}/user_requests.json", "w")

    all = 0
    format = 0
    for line in rf:
        all += 1
        data = json.loads(line)
        method = data["method"]
        n_tools = data["number_of_tools"]
        seed = data["seed"]
        _id = data["id"]
        sampled_nodes = data["sampled_nodes"]
        sampled_links = data["sampled_links"]
        for sampled_node in sampled_nodes:
            sampled_node["task"] = sampled_node["id"]
            sampled_node.pop("id")
            if dependency_type == "resource":
                sampled_node["task"] = sampled_node["task"].replace("_", " ")
                if "input" in sampled_node:
                    sampled_node["input-type"] = sampled_node["input"]
                    sampled_node.pop("input")
                    sampled_node["output-type"] = sampled_node["output"]
                    sampled_node.pop("output")
            else:
                sampled_node["arguments"] = sampled_node["parameters"]
                sampled_node.pop("parameters")
        user_request, task_steps, links, nodes, check_by_teacher = formulate_sample(data, dependency_type)
        if user_request is None:
            wf_error.write(line)
            continue
        format += 1
        result = {
            "id": _id,
            "seed": seed,
            "type": method,
            "n_tools": n_tools,
            "sampled_nodes": sampled_nodes,
            "sampled_links": sampled_links,
            "user_request": user_request,
            "task_steps": task_steps,
            "task_nodes": nodes,
            "task_links": links,
            "check_by_teacher": check_by_teacher,
        }
        wf_format.write(json.dumps(result)+"\n")
        ur_result = {
            "id": _id,
            "user_request": user_request,
        }
        wf_ur.write(json.dumps(ur_result)+"\n")
    wf_format.close()
    wf_error.close()
    wf_ur.close()
    rf.close()
    print(f"Format {format} out of {all}")

if __name__ == "__main__":
    formulate()