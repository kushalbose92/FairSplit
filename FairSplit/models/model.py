from models.gcn import GCN
from models.sage import SAGE
from models.appnp import APPNP
import torch
import torch.nn.functional as F
from load import split_graph_by_SH
from utils import compute_metrics

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
class FairSplit_model(torch.nn.Module):
    def __init__(self, param_dict, n_edges, n_non_edges):
        super(FairSplit_model, self).__init__()
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
        #self.lambda_raw = torch.nn.Parameter(torch.tensor(0.0))  # unconstrained

        # based on model choice in args, set up model - gcn / sage / appnp
        if param_dict['model_name'] == 'gcn':
            self.submodel = GCN(param_dict)
        elif param_dict['model_name'] == 'sage':
            self.submodel = SAGE(param_dict)
        elif param_dict['model_name'] == 'appnp':
            self.submodel = APPNP(param_dict, K=2, alpha=0.1)

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
        homophilic_feats = self.submodel(x, data.homophilic_edges)
        heterophilic_feats = self.submodel(x, data.heterophilic_edges)
        output = self.transformation(homophilic_feats, heterophilic_feats)
        return(output)
        
    def run_training(self, data):    
        self.train()
        print("...training ...  0%", end="", flush=True)
        CE_loss_hist, CF_loss_hist = [], []
        #lambda_hist = []
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

            #lambda_f = torch.nn.functional.softplus(self.lambda_raw)  # always > 0
            #loss = CE_loss + lambda_f * CF_loss
            loss = CE_loss + CF_loss
            loss.backward()
            #lambda_hist.append(lambda_f)
            self.optim.step()

        print("\b\b\b\b100%")
        #return(CE_loss_hist, CF_loss_hist, lambda_hist)
        return(CE_loss_hist, CF_loss_hist)

    def evaluate(self, data):
        self.eval()
        output = self.forward(data)
        if (output.argmax(dim=1).sum() == data.num_nodes) or (output.argmax(dim=1).sum() == 0):
            print("All predictions same")
        acc, f1_val, parity, equality = compute_metrics(data.y, output, data.sens, data.idx_val) 
        return(acc, f1_val, parity, equality)
       

