import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a graph with 10 nodes
G = nx.Graph()

# Define node colors: 0 (black), 1 (blue)
seed = 2
random.seed(seed)
n_nodes = 8
node_colors = {0: 'black', 1: 'blue'}
nodes = {i: random.choice([0, 1]) for i in range(n_nodes)}  # Randomly assign classes

# Add nodes to the graph
for node, cls in nodes.items():
    G.add_node(node, color=node_colors[cls])

# Generate more edges to ensure connectivity
edges = set()
while len(edges) < 20:
    u, v = random.sample(list(G.nodes), 2)
    if (u, v) not in edges and (v, u) not in edges:
        edges.add((u, v))

# Assign edge colors based on homophily
edge_colors = {}
for u, v in edges:
    if nodes[u] == nodes[v]:  # Homophilic (same class)
        edge_colors[(u, v)] = 'green'
    else:  # Heterophilic (different class)
        edge_colors[(u, v)] = 'red'
    G.add_edge(u, v)

# Create subgraphs while retaining edge colors
G_homo = nx.Graph()
G_hetero = nx.Graph()

for edge, color in edge_colors.items():
    if color == 'green':
        G_homo.add_edge(*edge, color=color)
    else:
        G_hetero.add_edge(*edge, color=color)

# Ensure all nodes exist in subgraphs
for node in G.nodes():
    G_homo.add_node(node, color=node_colors[nodes[node]])
    G_hetero.add_node(node, color=node_colors[nodes[node]])

# Random layout
pos = nx.spring_layout(G, seed=7)

# Get edge colors for original graph (Fixed KeyError)
edge_colors_list = [edge_colors.get((u, v), edge_colors.get((v, u), 'black')) for u, v in G.edges()]

# Increase figure width for better spacing
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot the original graph
axes[0].set_title("Original Graph", fontsize=28)
nx.draw(G, pos, ax=axes[0], with_labels=True, node_color=[node_colors[nodes[n]] for n in G.nodes()], 
        edge_color=edge_colors_list, width=2.5, node_size=500, font_color="white", font_weight="bold")

# Plot homophilic subgraph
axes[1].set_title("Homophilic Subgraph", fontsize=28)
nx.draw(G_homo, pos, ax=axes[1], with_labels=True, node_color=[node_colors[nodes[n]] for n in G_homo.nodes()], 
        edge_color=[G_homo[u][v]['color'] for u, v in G_homo.edges()], width=2.5, node_size=500, font_color="white", font_weight="bold")

# Plot heterophilic subgraph
axes[2].set_title("Heterophilic Subgraph", fontsize=28)
nx.draw(G_hetero, pos, ax=axes[2], with_labels=True, node_color=[node_colors[nodes[n]] for n in G_hetero.nodes()], 
        edge_color=[G_hetero[u][v]['color'] for u, v in G_hetero.edges()], width=2.5, node_size=500, font_color="white", font_weight="bold")

# Remove axis ticks
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust subplot spacing
fig.subplots_adjust(wspace=0.4, left=0.05, right=0.95)

# Add vertical divider lines
for ax in [axes[0], axes[1]]:
    ax.axvline(x=max(pos.values(), key=lambda x: x[0])[0] + 0.15, color='gray', linewidth=2)
plt.show()
