import torch
import torch.nn as nn
from torch_geometric.nn.models import LINKX

class LinkX(nn.Module):
    def __init__(self, param_dict, num_layers=2):
        super(LinkX, self).__init__()
        self.model_name = 'linkx'

        self.linkx = LINKX(
            in_channels=param_dict['n_features'],
            hidden_channels=param_dict['hidden'],
            out_channels=param_dict['hidden'],
            num_nodes=param_dict['n_nodes'],
            num_layers=num_layers,
            dropout=param_dict['dropout']
        )

        # Add final transformation layer if needed
        self.fc = nn.Linear(param_dict['hidden'], param_dict['hidden'])

        # Optional batch norm & dropout after LINKX
        self.bn = nn.BatchNorm1d(param_dict['hidden'])
        self.dropout = nn.Dropout(p=param_dict['dropout'])

        self.reset_parameters()

    def reset_parameters(self):
        self.linkx.reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.linkx(x, edge_index)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
