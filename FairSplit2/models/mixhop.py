import torch
import torch.nn as nn
from torch_geometric.nn import MixHopConv

class MixHop(torch.nn.Module):
    def __init__(self, param_dict, hops=[0, 1, 2]):
        super(MixHop, self).__init__()
        self.model_name = 'mixhop'

        self.mixhop1 = MixHopConv(
            in_channels=param_dict['n_features'],
            out_channels=param_dict['hidden'],
            powers=hops
        )
        self.bn1 = nn.BatchNorm1d(param_dict['hidden'] * len(hops))
        self.dropout1 = nn.Dropout(p=param_dict['dropout'])

        self.mixhop2 = MixHopConv(
            in_channels=param_dict['hidden'] * len(hops),
            out_channels=param_dict['hidden'],
            powers=hops
        )
        self.bn2 = nn.BatchNorm1d(param_dict['hidden'] * len(hops))
        self.dropout2 = nn.Dropout(p=param_dict['dropout'])

        self.fc = nn.Linear(param_dict['hidden'] * len(hops), param_dict['hidden'])

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.mixhop1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.mixhop2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc(x)
        return x
