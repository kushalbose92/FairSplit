import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE

class SAGE(nn.Module):
    def __init__(self, param_dict):
        super(SAGE, self).__init__()
        self.model_name = 'sage'

        # GraphSAGE without internal dropout
        self.sage = GraphSAGE(
            in_channels=param_dict['n_features'], 
            hidden_channels=param_dict['hidden'], 
            num_layers=2,  
            out_channels=param_dict['hidden'],  
            dropout=0.0,  # 🚨 Set dropout to 0 to disable internal dropout
            aggr='mean'  
        )

        # Transition layer (ReLU + BatchNorm + Dropout)
        self.batch_norm = nn.BatchNorm1d(param_dict['hidden'])
        self.dropout = nn.Dropout(p=param_dict['dropout'])  # Apply dropout outside instead

        # Fully connected layer for classification
        self.fc = nn.Linear(param_dict['hidden'], param_dict['hidden'])
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)  # GraphSAGE forward pass
        
        x = F.relu(x)  # Apply ReLU
        x = self.batch_norm(x)  # Apply BatchNorm
        x = self.dropout(x)  # Apply Dropout (only once)

        return self.fc(x)  # Final classification layer
