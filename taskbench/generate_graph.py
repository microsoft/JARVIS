import json
import click

def generate_graph_resource(tool_file):
    with open(tool_file) as f:
        data = json.load(f)
    data = data["nodes"]
    assert "input-type" in data[0] and "output-type" in data[0], "Input and output types are not defined"
    nodes = []
    for i in range(len(data)):
        nodes.append({"id": data[i]["id"], "desc": data[i]["desc"], "input-type": data[i]["input-type"], "output-type": data[i]["output-type"]})
    links = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                if len(set(nodes[i]["output-type"]).intersection(set(nodes[j]["input-type"]))) > 0:
                    links.append({"source": nodes[i]["id"], "target": nodes[j]["id"], "type": list(set(nodes[i]["output-type"]).intersection(set(nodes[j]["input-type"])))[0]})
    graph = {"nodes": nodes, "links": links}
    with open(tool_file.replace("tools", "graph"), 'w') as f:
        json.dump(graph, f, indent=2)

def generate_graph_temporal(tool_file):
    with open(tool_file) as f:
        data = json.load(f)
    nodes = []
    data = data["nodes"]
    if "parameters" not in data[0] and "input-type" not in data[0]:
        for i in range(len(data)):
            nodes.append({"id": data[i]["id"], "desc": data[i]["desc"]})
    elif "input-type" not in data[0]:
        for i in range(len(data)):
            nodes.append({"id": data[i]["id"], "desc": data[i]["desc"], "parameters": data[i]["parameters"]})
    else:
        for i in range(len(data)):
            nodes.append({"id": data[i]["id"], "desc": data[i]["desc"], "parameters": data[i]["parameters"], "input-type": data[i]["input-type"], "output-type": data[i]["output-type"]})
    links = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                links.append({"source": nodes[i]["id"], "target": nodes[j]["id"], "type": "complete"})
    graph = {"nodes": nodes, "links": links}
    with open(tool_file.replace("tools", "graph"), 'w') as f:
        json.dump(graph, f, indent=2)

@click.command()
@click.option('--data_dir')
@click.option('--tool_desc', default=None, help='Path to the tool description file')
@click.option('--dependency_type', default='resource', help='Type of graph to generate')
def generate_graph(tool_desc, data_dir, dependency_type):
    if tool_desc:
        tool_file = tool_desc
    else:
        tool_file = f"{data_dir}/graph_desc.json"
    if dependency_type == "temporal":
        generate_graph_temporal(tool_file)
    elif dependency_type == "resource":
        generate_graph_resource(tool_file)
    else:
        print("Type not supported")

if __name__ == "__main__":
    generate_graph()