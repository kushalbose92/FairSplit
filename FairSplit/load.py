"""
program: load script for fairness

"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def generate_edge_index(edges, n_nodes, add_self_loops=False):

    if add_self_loops:
        self_loops = torch.arange(n_nodes).unsqueeze(1)  # Diagonal indices for self-loops
        edges = torch.cat([edges, torch.stack([self_loops.squeeze(), self_loops.squeeze()], dim=1)], dim=0)
    
    # Step 2: Symmetrize the edges
    symmetric_edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)  # Add reverse edges
    
    # Step 3: Remove duplicate edges
    unique_edges = torch.unique(symmetric_edges, dim=0)
    
    # Step 4: Convert to COO format
    row, col = unique_edges[:, 0], unique_edges[:, 1]
    edge_index = torch.stack([row, col], dim=0)

    # Step 5: Sort edges on nodes    
    sorted_indices = torch.argsort(edge_index[0], stable=True)  # Stable sort ensures consistency
    edge_index = edge_index[:, sorted_indices]
    
    return edge_index

def generate_non_edge_index(n_nodes, edge_index, n_add_edges):
    # generate random points of non-edges, same number of edges
    # initially taken double count to address removal due to:
        # self-loops & duplicates & overlap with edges
    #n_edges = edge_index.shape[1]
    non_edge_index = np.random.randint(1, n_nodes, size=(2, int(n_add_edges*2.0)))
    valid_flags = non_edge_index[0]!=non_edge_index[1] # remove self-loops
    non_edge_index = non_edge_index[:, valid_flags]
    non_edge_index_set = set([tuple(t) for t in np.transpose(non_edge_index)])
    edge_index_set = set([tuple(t) for t in np.transpose(edge_index)])
    non_edge_index = list(non_edge_index_set - edge_index_set)
    non_edge_index = np.transpose(non_edge_index[:n_add_edges])
    #n_edges = non_edge_index.shape[0]
    # sort the index    
    lexsorted_indices = np.lexsort((non_edge_index[1, :], non_edge_index[0, :]))
    non_edge_index = non_edge_index[:, lexsorted_indices]
    return(torch.from_numpy(non_edge_index))

def load_dataset(dataset_name, data_folder, param_dict):
    # set dataset specific values e.g. gender
    sens_attr = param_dict['sens_attr']
    sens_idx = param_dict['sens_idx']
    predict_attr = param_dict['predict_attr']
    remove_cols = param_dict['remove_cols']
    
    # load features
    df = pd.read_csv(f"{data_folder}fairness/{dataset_name}/{dataset_name}.csv")
    header = list(df.columns)
    header.remove(predict_attr)
    header = [item for item in header if item not in remove_cols]

    # Sensitive Attribute
    if dataset_name=='German':
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    features = torch.tensor(df[header].values, dtype=torch.float32)
    labels = df[predict_attr].values
    if dataset_name=='German':
        labels[labels == -1] = 0
    labels = torch.LongTensor(labels)

    # load edges
    edges = np.loadtxt(f'{data_folder}fairness/{dataset_name}/{dataset_name}_edges.txt').astype('int')
    edges = edges[np.lexsort(edges.T)]
    edges = torch.from_numpy(edges).long()

    # create edge_index
    n_nodes = param_dict['n_nodes']
    edge_index = generate_edge_index(edges, n_nodes, add_self_loops=False)
    n_edges = edge_index.shape[1]
    
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    np.random.shuffle(label_idx_0)
    np.random.shuffle(label_idx_1)
    n0, n1 = len(label_idx_0), len(label_idx_1)

    idx_train = np.append(label_idx_0[:int(n0*0.8)], label_idx_1[:int(n1*0.8)])
    idx_val = np.append(label_idx_0[int(n0*0.8):int(n0*0.9)], label_idx_1[int(n1*0.8):int(n1*0.9)])
    idx_test = np.append(label_idx_0[int(n0*0.9):], label_idx_1[int(n1*0.9):])

    sens = df[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # randomlyh shuffle indices
    idx_train = idx_train[torch.randperm(idx_train.shape[0])]
    idx_val = idx_val[torch.randperm(idx_val.shape[0])]
    idx_test = idx_test[torch.randperm(idx_test.shape[0])]
    
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    norm_features = 2*(features - min_values).div(max_values-min_values) - 1
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features

    graph_data = Data(x=features, y=labels, edge_index=edge_index, idx_train=idx_train, 
                      idx_val=idx_val, idx_test=idx_test, sens=sens, 
                      num_nodes=n_nodes, num_edges=n_edges, num_non_edges=0)

    return graph_data
        
def dfs(graph_data):
    num_nodes = graph_data.num_nodes
    edge_index = graph_data.edge_index
    
    # Convert edge_index to adjacency list
    adj_list = {i: [] for i in range(num_nodes)}
    for src, dst in edge_index.t().tolist():
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # Assuming an undirected graph
    
    visited = set()
    dfs_order = []
    
    def dfs_visit(node):
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                dfs_order.append(curr)
                # Add neighbors in reverse order for consistent traversal
                stack.extend(sorted(adj_list[curr], reverse=True))
    
    # Ensure all components are visited
    for node in range(num_nodes):
        if node not in visited:
            dfs_visit(node)
    
    return dfs_order

# partition graph in subgraphs of a max number of nodes
def get_subgraphs(dataset_name, data, max_nodes, max_parts=10):
    
    device = data.x.device
    n_nodes = data.num_nodes
    #nodes = np.arange(n_nodes)
    nodes = dfs(data)
    np.random.shuffle(nodes)
    n_chunks = int(np.ceil(n_nodes/max_nodes))
    chunks = np.array_split(nodes, n_chunks)
    # remove last subgraph if too small
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks = chunks[:-1]
    if len(chunks) > max_parts:
        chunks = chunks[:max_parts]

    subgraphs = []
    chunk_count = 0 # may go upto max_parts
    for selected_nodes in chunks:
    
        selected_nodes = torch.tensor(sorted(selected_nodes), dtype=torch.long).to(device)
        selected_node_mappings = {orig_idx.item(): new_idx 
                          for new_idx, orig_idx in enumerate(selected_nodes)}
        
        # Extract the subgraph
        subgraph_edge_index = subgraph(selected_nodes, data.edge_index, 
                               relabel_nodes=True, num_nodes=data.num_nodes)[0]
        subgraph_n_nodes = len(selected_nodes)
        subgraph_n_edges = subgraph_edge_index.shape[1]

        # create non edge index
        subgraph_non_edge_index = generate_non_edge_index(subgraph_n_nodes, subgraph_edge_index, subgraph_n_edges)
        n_non_edges = subgraph_non_edge_index.shape[1]
        subgraph_edge_index = torch.cat((subgraph_edge_index, subgraph_non_edge_index), dim = 1)

        idx_train = data.idx_train[torch.isin(data.idx_train, selected_nodes)]
        idx_test = data.idx_test[torch.isin(data.idx_test, selected_nodes)]
        idx_val = data.idx_val[torch.isin(data.idx_val, selected_nodes)]

        idx_train = torch.tensor([selected_node_mappings.get(idx.item(), -1) for idx in idx_train])
        idx_train = idx_train[idx_train != -1]
        idx_test = torch.tensor([selected_node_mappings.get(idx.item(), -1) for idx in idx_test])
        idx_test = idx_test[idx_test != -1]
        idx_val = torch.tensor([selected_node_mappings.get(idx.item(), -1) for idx in idx_val])
        idx_val = idx_val[idx_val != -1]

        sens = data.sens[selected_nodes]
        subgraph_data = Data(x=data.x[selected_nodes], y=data.y[selected_nodes], 
                             edge_index=subgraph_edge_index, idx_train=idx_train, 
                             idx_test=idx_test, idx_val=idx_val, sens=sens, 
                             num_nodes=subgraph_n_nodes, num_edges=subgraph_n_edges, 
                             num_non_edges=n_non_edges)
        subgraphs.append(subgraph_data)
        chunk_count += 1
        if chunk_count>= max_parts:
            break
    del data
    return(subgraphs)

def add_random_edges(edge_index, num_nodes, m):
    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    
    new_edges = set()
    remaining = m  # Keep track of remaining edges to add

    while remaining > 0:
        # Generate candidate edges
        src = torch.randint(0, num_nodes, (remaining,))
        dst = torch.randint(0, num_nodes, (remaining,))

        # Filter out self-loops & existing edges
        candidates = {(s.item(), d.item()) for s, d in zip(src, dst) if s != d and (s.item(), d.item()) not in edge_set}

        # Add only up to the required number of edges
        new_edges.update(candidates)
        remaining = m - len(new_edges)  # Update the remaining count

    # Convert to tensor and concatenate
    new_edges_tensor = torch.tensor(list(new_edges)).T.to(edge_index.device)
    updated_edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)

    return updated_edge_index

# split a graph into to by s-homophily of edges
def split_graph_by_SH(data):

    edge_index = data.edge_index
    node_labels = data.sens  # Node labels

    # Get source and target nodes of edges
    src, tgt = edge_index.cpu().numpy()

    # Determine homophilic edges (same labels) and heterophilic edges (different labels)
    homophilic_mask = node_labels[src] == node_labels[tgt]
    heterophilic_mask = ~homophilic_mask

    # Split edge indices
    homophilic_edges = edge_index[:, homophilic_mask]
    heterophilic_edges = edge_index[:, heterophilic_mask]

    return homophilic_edges, heterophilic_edges

