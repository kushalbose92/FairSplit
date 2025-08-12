"""
program: utilities for affinity-based fairness
referred: affNet

"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score
import torch
import matplotlib.pyplot as plt
import ast

# set all seeds for reproducibility
def set_seeds(seed=13):
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Torch RNG
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Force deterministic algorithms (PyTorch >= 1.8)
    torch.use_deterministic_algorithms(True)

    # If using torch_geometric or DataLoader, also set:
    # worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)

def parse_arg(root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="D:/Indranil/ML2/Datasets/")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_homo', type=str, default='gcn')
    parser.add_argument('--model_hetero', type=str, default='gcn')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--n_nodes', type=int)
    parser.add_argument('--n_features', type=int)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--max_nodes', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    
    param_dict = {}

    param_dict['data_folder'] = args.data_folder
    param_dict['dataset_name'] = args.dataset
    param_dict['model_homo'] = args.model_homo
    param_dict['model_hetero'] = args.model_hetero
    param_dict['root'] = root
    param_dict['data_folder'] = args.data_folder
    param_dict['seed'] = args.seed
    param_dict['n_nodes'] = args.n_nodes
    param_dict['n_features'] = args.n_features
    param_dict['n_classes'] = args.n_classes
    param_dict['max_nodes'] = args.max_nodes
    param_dict['epochs'] = args.epochs    
    param_dict['init_lr'] = args.init_lr
    param_dict['weight_decay'] = args.weight_decay
    param_dict['hidden'] = args.hidden
    param_dict['dropout'] = args.dropout

    if args.dataset == 'Credit':
            param_dict['n_nodes'] = 30000
            param_dict['n_features'] = 13
            param_dict['n_classes'] = 2
            param_dict['emb_features'] = 13
            param_dict['sens_attr'] = "Age"
            param_dict['sens_attr_idx'] = 1
            param_dict['sens_idx'] = 1
            param_dict['predict_attr'] = 'NoDefaultNextMonth'
            param_dict['remove_cols'] = ['Single']
    elif args.dataset == 'German':
            param_dict['n_nodes'] = 1000
            param_dict['n_features'] = 27
            param_dict['n_classes'] = 2
            param_dict['emb_features'] = 27
            param_dict['sens_attr'] = "Gender"  
            param_dict['sens_attr_idx'] = 0
            param_dict['sens_idx'] = 0
            param_dict['predict_attr'] = "GoodCustomer" 
            param_dict['remove_cols'] = ['OtherLoansAtStore', 'PurposeOfLoan']
    elif args.dataset == 'Recidivism':
            param_dict['n_nodes'] = 18876
            param_dict['n_features'] = 18
            param_dict['n_classes'] = 2
            param_dict['emb_features'] = 18
            param_dict['sens_attr'] = "WHITE" 
            param_dict['sens_attr_idx'] = 0
            param_dict['sens_idx'] = 0
            param_dict['predict_attr'] = "RECID"
            param_dict['remove_cols'] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dict['device'] = device

    return(param_dict)

def assert_folders(root):
    results_folder = f"{root}results/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    hist_folder = f"{root}results/hist/"
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)
    return results_folder

def compute_metrics(y, output, sens, idx):
    yy = y[idx].cpu()
    oo = output[idx].cpu()
    ss = sens[idx].cpu()
    preds = oo.squeeze().cpu()
    preds = preds.argmax(axis=1)

    #accuracy and F1-score
    acc = torch.sum(yy==preds).item()/len(yy)
    f1_val = f1_score(yy, preds)
    
    #Parity & Equality 
    idx_s0 = ss==0
    idx_s1 = ss==1
    idx_s0_y1 = torch.bitwise_and(idx_s0, yy==1)
    idx_s1_y1 = torch.bitwise_and(idx_s1, yy==1)

    eps = 1e-8 # epsilon to avoid zero division
    equality = torch.abs(torch.sum(preds[idx_s0_y1])/(torch.sum(idx_s0_y1)+eps)-torch.sum(preds[idx_s1_y1])/(torch.sum(idx_s1_y1)+eps))
    parity = torch.abs(torch.sum(preds[idx_s0])/(torch.sum(idx_s0)+eps)-torch.sum(preds[idx_s1])/(torch.sum(idx_s1)+eps))

    return(acc, f1_val, parity, equality)

def get_structure_info(data):

    n_nodes = len(data.y)
    n_edges = data.edge_index.shape[1]

    # avg degree 
    avg_degree = n_edges/n_nodes

    # no of nodes with zero degree 
    active_nodes = len(torch.unique(data.edge_index))
    isolated_nodes = n_nodes - active_nodes

    # sensitivity-based homophily
    h_mask = (data.sens.unsqueeze(0) == data.sens.unsqueeze(1)).int()
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.int32).to(h_mask.device)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    n_homophilic_edges = torch.sum(adj.to(torch.int64) & h_mask.to(torch.int64)).item()
    h_ratio = n_homophilic_edges / n_edges
    
    return avg_degree, isolated_nodes, h_ratio

# detect outliers in grad of edges / non-edges
# mode is "add" for non-edges and "del" for edges
def detect_outliers(tensor, k, mode):
    mean = tensor.mean()
    std = tensor.std()   
    lower_bound = mean - k * std
    upper_bound = mean + k * std
    if mode=="add":
        outliers = (tensor < lower_bound)
    else:
        outliers = (tensor > upper_bound)
    return outliers 

def plot_hist(CE_loss_hist, CF_loss_hist, dataset_name, fname=None):
    plt.figure(figsize=(4,3), dpi=600)
    plt.subplot(1,2,1)
    plt.plot(CE_loss_hist, label="CE loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("classification loss")
    plt.subplot(1,2,2)
    plt.plot(CF_loss_hist, label="counterfactual loss")
    plt.xlabel("epochs")
    plt.title("fairness loss")
    plt.suptitle(dataset_name)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

