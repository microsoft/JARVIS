import networkx as nx
import random
import json
import matplotlib.pyplot as plt
import click

random.seed(0)
class GraphSampler:
    def __init__(self, graph: nx.Graph = None, file_name = None):
        if file_name:
            with open(file_name, "r") as f:
                data = json.load(f)

            # Represent your graph in NetworkX
            graph = nx.DiGraph()

            # Add nodes to the graph
            if "input-type" in data["nodes"][0]:
                for node in data["nodes"]:
                    graph.add_node(node["id"], desc=node["desc"], input_type=node["input-type"], output_type=node["output-type"])
            else:
                for node in data["nodes"]:
                    graph.add_node(node["id"], desc=node["desc"], parameters=node["parameters"])

            # Add edges to the graph
            for link in data["links"]:
                graph.add_edge(link["source"], link["target"], type=link["type"])

        self.graph = graph

    def sample_subgraph_by_weight(self, number_weights, method_weights):
        method = random.choices(list(method_weights.keys()), weights=list(method_weights.values()))[0]
        if method == "single":
            tool_number = 1
        else:
            tool_number = random.choices(list(number_weights.keys()), weights=list(number_weights.values()))[0]
        return self.sample_subgraph(tool_number, sample_method=method)

    def sample_subgraph(self, num_nodes=3, sample_method="chain"):
        seed_node = random.choice(list(self.graph.nodes))
        if sample_method == "single":
            sub_G = nx.DiGraph()
            sub_G.add_node(seed_node)
            return sub_G
        elif sample_method == "chain":
            return self.sample_subgraph_chain(seed_node, num_nodes)
        elif sample_method == "dag":
            return self.sample_subgraph_dag(seed_node, num_nodes)
        else:
            raise ValueError("Invalid sample method")

    def sample_subgraph_chain(self, seed_node, num_nodes):
        # Create a list to store the sub-graph nodes
        sub_graph_nodes = [seed_node]
        head_node = seed_node
        tail_node = seed_node
        edges = []

        # Keep adding nodes until we reach the desired number
        while len(sub_graph_nodes) < num_nodes:
            # Get the neighbors of the last node in the sub-graph
            head_node_neighbors = list(self.graph.predecessors(head_node))
            tail_node_neighbors = list(self.graph.successors(tail_node))
            neighbors = head_node_neighbors + tail_node_neighbors

            # If the node has neighbors, randomly select one and add it to the sub-graph
            if len(neighbors) > 0:
                neighbor = random.choice(neighbors)
                if neighbor not in sub_graph_nodes:
                    if neighbor in head_node_neighbors:
                        sub_graph_nodes.insert(0, neighbor)
                        edges.insert(0, (neighbor, head_node))
                        head_node = neighbor
                    else:
                        sub_graph_nodes.append(neighbor)
                        edges.append((tail_node, neighbor))
                        tail_node = neighbor
            else:
                break

        # Create the sub-graph
        sub_G = nx.DiGraph()
        sub_G.add_nodes_from(sub_graph_nodes)
        sub_G.add_edges_from(edges)

        return sub_G

    def sample_subgraph_dag(self, seed_node, num_nodes):
        # Create a list to store the sub-graph nodes
        sub_graph_nodes = [seed_node]
        edges = []

        # Keep adding nodes until we reach the desired number
        while len(sub_graph_nodes) < num_nodes:
            # Randomly select a node from the current sub-graph
            node = random.choice(sub_graph_nodes)
            # prec_neighbors = list(self.graph.predecessors(node))
            succ_neighbors = list(self.graph.successors(node))

            if "input_type" in self.graph.nodes[node]:
                # filter exisiting income edge type
                prec_neighbors = []
                input_type = list(self.graph.nodes[node]["input_type"])
                all_in_edges = list(self.graph.in_edges(node, data=True))
                for edge in edges:
                    for ref_edge in all_in_edges:
                        if edge[0] == ref_edge[0] and edge[1] == ref_edge[1]:
                            input_type.remove(ref_edge[2]["type"])
                for edge in all_in_edges:
                    if edge[2]["type"] in input_type:
                        prec_neighbors.append(edge[0])
            else:
                prec_neighbors = list(self.graph.predecessors(node))

            neighbors = prec_neighbors + succ_neighbors

            # If the node has neighbors, randomly select one and add it to the sub-graph
            if neighbors:
                neighbor = random.choice(neighbors)
                if neighbor not in sub_graph_nodes:
                    if neighbor in prec_neighbors:
                        edges.append((neighbor, node))
                    else:
                        edges.append((node, neighbor))
                    sub_graph_nodes.append(neighbor)
            # If the node has no neighbors, select a new node from the original graph
            else:
                node = random.choice(list(self.graph.nodes))
                if node not in sub_graph_nodes:
                    sub_graph_nodes.append(node)

        # Create the sub-graph
        sub_G = nx.DiGraph()
        sub_G.add_nodes_from(sub_graph_nodes)
        sub_G.add_edges_from(edges)

        return sub_G
    
    def sample_subgraph_random_walk(self, seed_node, num_nodes):
        # Create a list to store the sub-graph nodes
        sub_graph_nodes = [seed_node]
        edges = []

        # Keep adding nodes until we reach the desired number
        while len(sub_graph_nodes) < num_nodes:
            # Randomly select a node from the current sub-graph
            node = random.choice(sub_graph_nodes)
            neighbors = list(self.graph.successors(node))

            # If the node has neighbors, randomly select one and add it to the sub-graph
            if neighbors:
                neighbor = random.choice(neighbors)
                if neighbor not in sub_graph_nodes:
                    edges.append((node, neighbor))
                    sub_graph_nodes.append(neighbor)
            # If the node has no neighbors, select a new node from the original graph
            else:
                node = random.choice(list(self.graph.nodes))
                if node not in sub_graph_nodes:
                    sub_graph_nodes.append(node)

        # Create the sub-graph
        sub_G = nx.DiGraph()
        sub_G.add_nodes_from(sub_graph_nodes)
        sub_G.add_edges_from(edges)

        return sub_G
    
    def sample_subgraph_random_walk_with_restart(self, seed_node, num_nodes, restart_prob=0.15):
        # Create a list to store the sub-graph nodes
        sub_graph_nodes = [seed_node]
        edges = []

        # Keep adding nodes until we reach the desired number
        while len(sub_graph_nodes) < num_nodes:
            # Randomly select a node from the current sub-graph
            node = random.choice(sub_graph_nodes)
            neighbors = list(self.graph.successors(node))

            # If the node has neighbors, randomly select one and add it to the sub-graph
            if neighbors:
                neighbor = random.choice(neighbors)
                if neighbor not in sub_graph_nodes:
                    edges.append((node, neighbor))
                    sub_graph_nodes.append(neighbor)
            # If the node has no neighbors, select a new node from the original graph
            else:
                node = random.choice(list(self.graph.nodes))
                if node not in sub_graph_nodes:
                    sub_graph_nodes.append(node)
            
            # Randomly restart the walk
            if random.random() < restart_prob:
                node = random.choice(list(self.graph.nodes))
                if node not in sub_graph_nodes:
                    sub_graph_nodes.append(node)

        # Create the sub-graph
        sub_G = nx.DiGraph()
        sub_G.add_nodes_from(sub_graph_nodes)
        sub_G.add_edges_from(edges)

        return sub_G

@click.command()
@click.option('--file_name', default='graph_desc_original.json', help='Path to the json file')
@click.option('--sample_method', default='chain', help='Type of graph to generate')
@click.option('--num_nodes', default=3, help='Number of nodes in the subgraph')
@click.option('--save_figure', default=False, help='Save the figure')
def sample_subgraph(file_name, sample_method, num_nodes, save_figure):
    # Create a graph sampler
    random.seed(0)
    sampler = GraphSampler(file_name=file_name)

    # Sample a sub-graph
    sub_G = sampler.sample_subgraph(num_nodes, sample_method=sample_method)
    print("Sub-graph nodes:", sub_G.nodes)
    print("Sub-graph edges:", sub_G.edges)

    # Visualize the sub-graph
    if save_figure:
        pos = nx.circular_layout(sub_G)
        nx.draw_networkx_nodes(sub_G, pos, node_color="skyblue", node_size=300)
        nx.draw_networkx_edges(sub_G, pos, arrows=True)
        nx.draw_networkx_labels(sub_G, pos, font_size=8)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("test.png")



if __name__ == "__main__":
    sample_subgraph()