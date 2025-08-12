from models.gcn import GCN
from models.sage import SAGE
from models.appnp import APPNP
from models.mixhop import MixHop
from models.linkx import LinkX
import torch
import torch.nn.functional as F
from load import split_graph_by_SH
from utils import compute_metrics

def get_submodel(model_name, param_dict):
    if model_name == 'gcn':
        submodel = GCN(param_dict)
    elif model_name == 'sage':
        submodel = SAGE(param_dict)
    elif model_name == 'appnp':
        submodel = APPNP(param_dict, K=2, alpha=0.1)
    elif model_name == 'mixhop':
        submodel = MixHop(param_dict)
    elif model_name == 'linkx':
        submodel = LinkX(param_dict)
    return(submodel)

class ProjectionSummation(torch.nn.Module):
    def __init__(self, hidden, n_classes):
        super(ProjectionSummation, self).__init__()
        self.W1 = torch.nn.Linear(hidden, hidden, bias=False)  # Projection for h1
        self.W2 = torch.nn.Linear(hidden, hidden, bias=False)  # Projection for h2
        self.bn = torch.nn.BatchNorm1d(hidden)  # BatchNorm after Linear
        self.output_layer = torch.nn.Linear(hidden, n_classes)  # Final transformation to n dimensions

    def forward(self, h1, h2):
        combined = self.W1(h1) + self.W2(h2)  # Project and sum
        combined = self.bn(combined)
        output = self.output_layer(combined)  # Final projection to n-dim
        return output

# Define a model that uses the custom layers
class FairSplit2_model(torch.nn.Module):
    def __init__(self, param_dict, n_edges, n_non_edges):
        super(FairSplit2_model, self).__init__()
        self.dataset_name = param_dict['dataset_name']
        self.n_nodes = param_dict['n_nodes']
        self.n_features = param_dict['n_features']
        self.n_edges = n_edges
        self.n_non_edges = n_non_edges
        self.sens_attr_idx = param_dict['sens_attr_idx']
        self.epochs = param_dict['epochs']
        self.device = param_dict['device']
        self.lr=param_dict['init_lr']
        self.weight_decay=param_dict['weight_decay']

        # based on model choice in args, set up submodels - gcn / sage / appnp
        self.submodel_homo = get_submodel(param_dict['model_homo'], param_dict)
        self.submodel_hetero = get_submodel(param_dict['model_hetero'], param_dict)

        # define the projection-summation layer 
        self.transformation = ProjectionSummation(param_dict['hidden'], param_dict['n_classes'])
        
        # define the optimizer
        self.optim = torch.optim.Adam(self.parameters(), 
                     lr=self.lr, weight_decay=self.weight_decay)
        
    def forward(self, data, CF=False):
        if CF:
            x = data.counter_x
        else:
            x = data.x
        homophilic_feats = self.submodel_homo(x, data.homophilic_edges)
        heterophilic_feats = self.submodel_hetero(x, data.heterophilic_edges)
        output = self.transformation(homophilic_feats, heterophilic_feats)
        return(output)
        
    def run_training(self, data):    
        self.train()
        print("...training ...  0%", end="", flush=True)
        CE_loss_hist, CF_loss_hist = [], []
        data.counter_x = data.x.clone()
        data.counter_x[self.sens_attr_idx] = 1 - data.counter_x[self.sens_attr_idx]
        data.homophilic_edges, data.heterophilic_edges = split_graph_by_SH(data)

        for epoch in range(self.epochs):
            if epoch%100==0: 
                perc_completion = int(100*epoch/self.epochs)
                print(f"\b\b\b\b{perc_completion:>3}%", end="", flush=True)
    
            self.optim.zero_grad()

            # classification cross-entropy (CE) loss 
            output = self.forward(data)
            CE_loss = F.cross_entropy(output[data.idx_train], 
                             data.y[data.idx_train].to(self.device))
            CE_loss_hist.append(CE_loss.item())

            # counter-factual (CF) loss
            output = self.forward(data, CF=False)
            output_cf = self.forward(data, CF=True)
            CF_loss = F.cross_entropy(output_cf, output.argmax(dim=1))
            CF_loss_hist.append(CF_loss.item())

            loss = CE_loss + CF_loss
            loss.backward()
            self.optim.step()

        print("\b\b\b\b100%")
        return(CE_loss_hist, CF_loss_hist)

    def evaluate(self, data):
        self.eval()
        output = self.forward(data)
        if (output.argmax(dim=1).sum() == data.num_nodes) or (output.argmax(dim=1).sum() == 0):
            print("All predictions same")
        acc, f1_val, parity, equality = compute_metrics(data.y, output, data.sens, data.idx_val) 
        return(acc, f1_val, parity, equality)
       

