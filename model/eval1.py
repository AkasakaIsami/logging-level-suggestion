import configparser
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader

from dataset import MyDataset
from util import float_to_percent


class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_layers = 1
        self.num_directions = 1  # 单向LSTM
        self.hidden_size = 64

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(64, 5)
        self.act = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)

        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        pred = self.act(pred)
        return output[:, -1, :], pred


def idx2index(idx: torch.Tensor) -> torch.Tensor:
    """
    根据稀疏矩阵求index
    """
    index = []
    size = idx.shape[0]
    for i in range(size):
        if idx[i].item() == 1:
            index.append(i)
    return torch.tensor(index).long()


def visual(x, y, epoch):
    # t-SNE的最终结果的降维与可视化
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, )
    x = tsne.fit_transform(x)
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(x[:, 0], x[:, 1], c=y, label="t-SNE")
    # plt.legend()

    for i in range(x.shape[0]):
        plt.text(x[i, 0],
                 x[i, 1],
                 str(y[i]),
                 color=plt.cm.Set1(y[i]))

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    f = plt.gcf()  # 获取当前图像
    if epoch == -1:
        f.savefig(f'./result/test.png')
    else:
        f.savefig(f'./result/{epoch}.png')

    f.clear()  # 释放内存
    plt.show()


def transact(tensor: torch.Tensor) -> torch.Tensor:
    """
    把预测值转换成ordinal vector
    """
    size = tensor.shape[0]
    result = torch.zeros(size, 5)
    for i in range(size):
        for j in range(5):
            if tensor[i][j] > 0.5:
                result[i][j] = 1
            else:
                break
    return result


if __name__ == '__main__':
    """
    完成实验1：AST pooling + RNN (只到当前语句)
    """

    # 第一步：训练配置
    project = 'kafkademo'
    BS = 10
    LR = 1e-4
    EPOCHS = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 第二步 读取数据集
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    root_dir = cf.get('data', 'dataDir')
    root_dir = os.path.join(root_dir, project)
    dataset = MyDataset(root=root_dir, project=project)
    methods_info = pd.read_pickle(os.path.join(root_dir, 'processed', 'method_info.pkl'))

    # 第三步 切分数据集
    ratio = cf.get('data', 'ratio')
    ratios = [int(r) for r in ratio.split(':')]
    train_len = int(ratios[0] / sum(ratios) * len(dataset))
    val_len = int(ratios[1] / sum(ratios) * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset,
                                                            lengths=[train_len, val_len, test_len],
                                                            generator=torch.Generator().manual_seed(0))


    # 第四步 定义数据获取batch格式
    def my_collate_fn(batch):
        xs = []
        ys = []
        ids = []

        for data in batch:
            method = data.id
            info = methods_info.loc[methods_info['id'] == method]

            asts = info['ASTs'].tolist()[0]

            seq = torch.randn(0, 128)
            index = idx2index(data.idx).item()
            for i in range(index + 1):
                ast = asts[i].x
                ast = ast.mean(axis=0)
                ast = ast.reshape(1, 128)
                seq = torch.cat([seq, ast], dim=0)

            y = data.y
            y = y.reshape(1, y.shape[0])

            xs.append(seq)
            ys.append(y)
            ids.append(data.id + '@' + str(data.line.item()))
        xs = pad_sequence(xs, batch_first=True)
        ys = torch.cat([y for y in ys], dim=0).float()

        return xs, ys, ids


    # 第五步 获取数据加载器
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=my_collate_fn,
                              batch_size=BS,
                              shuffle=True)

    val_loader = DataLoader(dataset=val_dataset,
                            collate_fn=my_collate_fn,
                            batch_size=BS,
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=my_collate_fn,
                             batch_size=BS,
                             shuffle=True)

    # 第六步 训练准备
    model = MyLSTM().to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = torch.nn.MSELoss().to(device)

    # 定义用于评估预测结果的东西
    best_acc = 0.0
    best_model = model

    # 定义控制日志打印的东西
    total_train_step = 0

    # 第七步 正式开始训练
    for epoch in range(EPOCHS):
        print(f'------------第 {epoch + 1} 轮训练开始------------')

        # 训练
        model.train()
        for i, (x, y, ids) in enumerate(train_loader):
            model.zero_grad()
            _, y_hat = model(x.to(device))
            y = y.to(device)
            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

        # 验证
        total_val_loss = 0.0
        y_hat_total = torch.randn(0).to(device)
        y_total = torch.randn(0).to(device)

        xs = torch.randn(0, 64)
        ys = []

        model.eval()
        with torch.no_grad():
            for i, (x, y, ids) in enumerate(val_loader):
                h, y_hat = model(x.to(device))
                y = y.to(device)
                loss = loss_function(y_hat, y)

                # 用来计算整体指标
                total_val_loss += loss.item()

                y_hat = transact(y_hat).to(device)

                y_hat_total = torch.cat([y_hat_total, y_hat])
                y_total = torch.cat([y_total, y])

                # 根据实际lable
                size = len(y)
                for j in range(size):
                    label = torch.index_select(y.cpu(), dim=0, index=torch.tensor([j]))
                    temph = torch.index_select(h.cpu(), dim=0, index=torch.tensor([j]))
                    xs = torch.cat([xs, temph], dim=0)

                    if torch.equal(label, torch.tensor([[1, 1, 1, 1, 1]]).float()):
                        ys.append(4)
                    elif torch.equal(label, torch.tensor([[1, 1, 1, 1, 0]]).float()):
                        ys.append(3)
                    elif torch.equal(label, torch.tensor([[1, 1, 1, 0, 0]]).float()):
                        ys.append(2)
                    elif torch.equal(label, torch.tensor([[1, 1, 0, 0, 0]]).float()):
                        ys.append(1)
                    elif torch.equal(label, torch.tensor([[1, 0, 0, 0, 0]]).float()):
                        ys.append(0)

        print(f"验证集整体Loss: {total_val_loss}")

        acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
        print(f"验证集 accuracy_score: {float_to_percent(acc)}")

        if acc > best_acc:
            print(f"***当前模型的平衡准确率表现最好，被记为表现最好的模型***\n")
            best_model = model
            best_acc = acc

        print(f"***保存tsne中***\n")

        xs = xs.numpy()
        ys = np.array(ys)

        visual(xs, ys, epoch + 1)
