import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, BatchNorm, GATConv, global_max_pool, RGCNConv, RGATConv

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

        self.conv_0 = GCNConv(128, 64)
        self.bn_0 = BatchNorm(64)
        self.conv_1 = GCNConv(64, 64)
        self.bn_1 = BatchNorm(64)
        self.mlp = Linear(64, 5, weight_initializer='kaiming_uniform')

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        h = F.leaky_relu_(self.conv_0(x, edge_index))
        h = self.bn_0(h)
        h = F.leaky_relu_(self.conv_1(h, edge_index))
        h = self.bn_1(h)

        idx = data.idx
        h = torch.index_select(h, dim=0, index=idx2index(idx).to(self.device))
        out = torch.sigmoid(self.mlp(h))
        return h, out


class MyOutGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv_0 = GATConv(in_channels=128, out_channels=64, heads=2)
        self.bn_0 = BatchNorm(128)
        self.conv_1 = GATConv(in_channels=128, out_channels=64, heads=1)
        self.bn_1 = BatchNorm(64)

        self.mlp = nn.Sequential(Linear(64, 32, weight_initializer='kaiming_uniform'),
                                 nn.LeakyReLU(),
                                 Linear(32, 16, weight_initializer='kaiming_uniform'),
                                 nn.LeakyReLU(),
                                 Linear(16, 5, weight_initializer='kaiming_uniform'),
                                 )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        h = F.leaky_relu_(self.conv_0(x, edge_index))
        h = self.bn_0(h)
        h = F.leaky_relu_(self.conv_1(h, edge_index))
        h = self.bn_1(h)

        idx = data.idx
        h = torch.index_select(h, dim=0, index=idx2index(idx).to(self.device))
        h = self.mlp(h)
        out = torch.sigmoid(h)
        return h, out


class MyOutRGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        in_channels = 128
        hidden_channels = 128
        out_channels = 5

        self.conv_0 = RGCNConv(in_channels, hidden_channels, num_relations=2)
        self.conv_1 = RGCNConv(hidden_channels, hidden_channels, num_relations=2)
        self.mlp = Linear(hidden_channels, out_channels, weight_initializer='kaiming_uniform')

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        h = F.leaky_relu_(self.conv_0(x, edge_index, edge_type))
        h = F.leaky_relu_(self.conv_1(h, edge_index, edge_type))

        idx = data.idx
        h = torch.index_select(h, dim=0, index=idx2index(idx).to(self.device))
        out = torch.sigmoid(self.mlp(h))
        return h, out


class MyOutRGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        in_channels = 128
        hidden_channels = 128
        out_channels = 5

        self.conv_0 = RGATConv(128, 128, num_relations=2, heads=2)
        self.bn_0 = BatchNorm(2 * 128)
        self.conv_1 = RGATConv(2 * 128, 128, num_relations=2, heads=1)
        self.bn_1 = BatchNorm(128)
        self.mlp = Linear(hidden_channels, out_channels, weight_initializer='kaiming_uniform')

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        h = F.leaky_relu_(self.conv_0(x, edge_index, edge_type))
        h = self.bn_0(h)
        h = F.leaky_relu_(self.conv_1(h, edge_index, edge_type))
        h = self.bn_1(h)

        idx = data.idx
        h = torch.index_select(h, dim=0, index=idx2index(idx).to(self.device))
        out = torch.sigmoid(self.mlp(h))
        return h, out


class MyInGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        in_channels = 128
        hidden_channels = 64
        out_channels = 32

        self.conv_0 = GCNConv(in_channels, hidden_channels)
        self.conv_1 = GCNConv(hidden_channels, out_channels)
        self.mlp = Linear(hidden_channels, out_channels, weight_initializer='kaiming_uniform')

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index.long()
        batch = data.batch

        h = F.leaky_relu_(self.conv_0(x, edge_index))
        h = F.leaky_relu_(self.conv_1(h, edge_index))
        out = global_max_pool(h, batch)

        return out


class MyNestingOutGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = MyInGCN()
        self.conv_0 = GCNConv(32, 32)
        self.bn_0 = BatchNorm(32)
        self.conv_1 = GCNConv(32, 16)
        self.bn_1 = BatchNorm(16)
        self.mlp = Linear(16, 5, weight_initializer='kaiming_uniform')

    def forward(self, astss, data):
        statements_vec = self.encoder(astss)

        x = statements_vec
        edge_index = data.edge_index

        h = F.leaky_relu_(self.conv_0(x, edge_index))
        h = self.bn_0(h)
        h = F.leaky_relu_(self.conv_1(h, edge_index))
        h = self.bn_1(h)

        idx = data.idx
        h = torch.index_select(h, dim=0, index=idx2index(idx).to(self.device))
        out = torch.sigmoid(self.mlp(h))
        return h, out
