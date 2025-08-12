import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNP_base

class APPNP(torch.nn.Module):
    def __init__(self, param_dict, K=2, alpha=0.1):
        super(APPNP, self).__init__()
        self.model_name = 'appnp'
        self.lin1 = torch.nn.Linear(param_dict['n_features'], param_dict['hidden'])
        self.bn1 = torch.nn.BatchNorm1d(param_dict['hidden'])  # BatchNorm after Linear
        self.lin2 = torch.nn.Linear(param_dict['hidden'], param_dict['hidden'])
        self.bn2 = torch.nn.BatchNorm1d(param_dict['hidden'])  # BatchNorm after Linear
        self.prop1 = APPNP_base(K, alpha)
        self.dropout = param_dict['dropout']

    def reset_parameters(self):
        self.prop1.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return x
