"""
program: fairness with homophily based edge-splitting (FairSplit)

"""

# set environment
import sys, os
root = os.getcwd()+'/'
sys.path.append(root)

# import libraries
from utils import set_seeds, parse_arg, assert_folders
from utils import get_structure_info, plot_hist
from load import load_dataset, get_subgraphs
from models.model import FairSplit_model

import pandas as pd
import numpy as np
import gc

# create dataframe to store results
results_folder = assert_folders(root)
result_cols = ['dataset', 'model', 'init_lr', 'weight_decay', 'hiddden', 'dropout', 
               'epochs', 'avg_degree', 'isolated_nodes', 'h_ratio', 
               'acc', 'F1', 'D_sp', 'D_eo', 'acc_std', 'F1_std', 'D_sp_std', 'D_eo_std']
try:
   results_df = pd.read_csv(results_folder+'results.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

# set seed
seed = 13
set_seeds(seed)

# set up basic parameters
param_dict = parse_arg(root) # arguments passed thru commandline
model_name = param_dict['model_name']
dataset_name = param_dict['dataset_name']
data_folder = param_dict['data_folder']

# load dataset 
graph_data = load_dataset(dataset_name, data_folder, param_dict)
param_dict['orig_n_nodes'] = graph_data.num_nodes

# split large graphs into subgrpahs of fixed size
subgraphs = get_subgraphs(dataset_name, graph_data, param_dict['max_nodes'], max_parts=5)
del graph_data

run_log = []
for run in range(5):
    
    n_chunks = len(subgraphs)
    acc_list, f1_list, parity_list, equality_list = [], [], [], []
    avg_degree_list, isolated_nodes_list, h_ratio_list = [], [], []
    i = 0
    for sg in subgraphs:
        i += 1
        n_nodes, n_edges = sg.num_nodes, sg.num_edges
        param_dict['n_nodes'] = n_nodes
        sg = sg.to(param_dict['device'])
    
        model = FairSplit_model(param_dict, sg.num_edges, sg.num_non_edges)
        model = model.to(param_dict['device'])
        
        # run aff_trainer - standard / brute-force / fairedit
        #task_loss_hist, fairness_loss_hist, lambda_hist = model.run_training(sg) 
        task_loss_hist, fairness_loss_hist = model.run_training(sg) 
        if run==0 and i==1: # plot training history only for first subgraph
            fname = f"{root}results/hist/{dataset_name}_{param_dict['model_name']}"
            fname = fname+f"_{param_dict['init_lr']}_{param_dict['weight_decay']}"
            fname = fname+f"_{param_dict['dropout']}.png"
            plot_hist(task_loss_hist, fairness_loss_hist, dataset_name, fname)
        acc, f1_val, parity, equality = model.evaluate(sg)
        avg_degree, isolated_nodes, h_ratio = get_structure_info(sg)
        acc_list.append(acc)
        f1_list.append(f1_val)
        parity_list.append(parity.item())
        equality_list.append(equality.item())
        avg_degree_list.append(avg_degree)
        isolated_nodes_list.append(isolated_nodes)
        h_ratio_list.append(h_ratio)
        gc.collect()
    
        print(f"chunk {i}/{n_chunks}. ACC: {acc:.4f}, F1: {f1_val:.4f}, D_sp: {parity:.6f}, D_eo: {equality:.6f}")
        print(f"   num_edges: {sg.num_edges}, avg deg: {avg_degree:.1f}, 0-nodes: {isolated_nodes}, h: {h_ratio:.2f}")
    
    acc_val = np.mean(acc_list)
    f1_val = np.mean(f1_list)
    parity = np.mean(parity_list)
    equality = np.mean(equality_list)
    avg_degree = np.mean(avg_degree_list)
    isolated_nodes = np.mean(isolated_nodes_list)
    h_ratio = np.mean(h_ratio_list)
    
    run_log.append([avg_degree, isolated_nodes, h_ratio, acc, f1_val, parity, equality])

run_log = pd.DataFrame(run_log, columns=['avg_degree', 'isolated_nodes', 'h_ratio', 'acc', 'F1', 'D_sp', 'D_eo'])
_, _, _, acc, f1_val, parity, equality = run_log.mean(axis=0)
_, _, _, acc_std, f1_val_std, parity_std, equality_std = run_log.std(axis=0)

results_df.loc[len(results_df)] = [dataset_name, model_name, param_dict['init_lr'], 
                               param_dict['weight_decay'], param_dict['hidden'], 
                               param_dict['dropout'], param_dict['epochs'],  
                               avg_degree, isolated_nodes, h_ratio, 
                               acc, f1_val, parity, equality, 
                               acc_std, f1_val_std, parity_std, equality_std]
results_df.to_csv(results_folder+'results.csv', index=False, float_format="%.5f")
