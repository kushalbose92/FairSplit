import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, param_dict):
        super(GCN, self).__init__()
        self.model_name = 'gcn'
        self.gcn1 = GCNConv(param_dict['n_features'], param_dict['hidden'])
        self.bn1 = torch.nn.BatchNorm1d(param_dict['hidden'])  # BatchNorm after Linear
        self.dropout1 = nn.Dropout(p=param_dict['dropout'])  
        self.gcn2 = GCNConv(param_dict['hidden'], param_dict['hidden'])
        self.bn2 = torch.nn.BatchNorm1d(param_dict['hidden'])  # BatchNorm after Linear
        self.dropout2 = nn.Dropout(p=param_dict['dropout'])  
        self.fc = nn.Linear(param_dict['hidden'], param_dict['hidden'])

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        return x






