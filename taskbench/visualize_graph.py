import json
import networkx as nx
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('--data_dir')
def visialize_graph(data_dir):
    graph_file = f"{data_dir}/graph_desc.json"
    with open(graph_file, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for node in data["nodes"]:
        G.add_node(node["id"])

    for link in data["links"]:
        G.add_edge(link["source"], link["target"])

    pos = nx.spring_layout(G)
    pos = nx.random_layout(G)    
    pos = nx.kamada_kawai_layout(G)
    
    # Show the visualization
    plt.figure(figsize=(60, 60), dpi=80)
    plt.tight_layout()
    plt.axis("off")
    plt.show()

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1200)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=40)
    nx.draw_networkx_labels(G, pos, font_size=50, font_color="green", font_weight="bold")
    plt.savefig(graph_file.replace(".json", ".pdf"))

if __name__ == "__main__":
    visialize_graph()