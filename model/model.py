import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

from util import idx2index


class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_layers = 1
        self.num_directions = 1  # 单向LSTM
        self.hidden_size = 64

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(64, 5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(
            self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(
            self.device)

        output, _ = self.lstm(x, (h_0.detach(), c_0.detach()))
        output = self.dropout(output)
        pred = self.linear(output[:, -1, :])
        pred = self.act(pred)
        return output[:, -1, :], pred


class MyBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_layers = 2
        self.num_directions = 2  # 双向LSTM
        self.hidden_size = 64

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=self.num_layers, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(128, 5)
        self.act = nn.Sigmoid()

    def forward(self, x, idxs):
        batch_size, seq_len = x.shape[0], x.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(
            self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(
            self.device)

        h, _ = self.lstm(x, (h_0, c_0))

        temp = torch.randn(0, 128).to(self.device)
        for i in range(batch_size):
            select = torch.index_select(h[i], dim=0, index=torch.tensor(idxs[i]).to(self.device))
            temp = torch.cat([temp, select], dim=0)

        h = temp
        h = self.dropout(h)
        record = h
        h = self.linear(h)
        out = self.act(h)

        return record, out


class MyOutGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        in_channels = 128
        hidden_channels = 64
        out_channels = 5

        self.conv_0 = GCNConv(in_channels, hidden_channels)
        self.conv_1 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = Linear(hidden_channels, out_channels, weight_initializer='kaiming_uniform')

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        h=x
        try:
            h = F.leaky_relu_(self.conv_0(x, edge_index))
            h = F.leaky_relu_(self.conv_1(h, edge_index))
        except:
            print('hi')

        idx = data.idx
        h = torch.index_select(h, dim=0, index=idx2index(idx))
        out = torch.sigmoid(self.mlp(h))
        return h, out
