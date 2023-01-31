import configparser
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import random_split, DataLoader
from torch_geometric.data import Data, Batch

from dataset import MyDataset
from model import MyOutRGAT
from util import float_to_percent, transact, OR2OEN, AOD, visual, tensor2label, class_acc

"""
完成实验5：AST pooling + GNN (多边CFG+DFG的RGCN)
"""

if __name__ == '__main__':

    # 第一步：训练配置
    project = 'kafka'
    BS = 15
    LR = 5e-3
    EPOCHS = 100
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
    print(
        f"数据集切分完成，总共{len(dataset)}条数据，其中训练集{len(train_dataset)}条，验证集{len(val_dataset)}条，测试集{len(test_dataset)}条，")


    # 第四步 定义数据获取batch格式
    def my_collate_fn(batch):
        new_datalist = []

        for data in batch:
            method = data.id.split('@')[0]
            info = methods_info.loc[methods_info['id'] == method]

            x = torch.randn(0, 128)
            for ast in info['ASTs'].tolist()[0]:
                ast.x = ast.x.mean(axis=0)
                ast.x = ast.x.reshape(1, 128)
                ast.edge_index = torch.zeros(2, 0).long()
                x = torch.cat([x, ast.x], dim=0)

            cfg_edge_index = info['edges'].tolist()[0][0].long()
            dfg_edge_index = info['edges'].tolist()[0][1].long()

            y = data.y
            y = y.reshape(1, y.shape[0])

            edge_index = torch.cat([cfg_edge_index, dfg_edge_index], 1)
            len_1 = cfg_edge_index.shape[1]
            len_2 = dfg_edge_index.shape[1]
            edge_type_1 = torch.zeros(len_1, )
            edge_type_2 = torch.ones(len_2, )
            edge_type = torch.cat([edge_type_1, edge_type_2], -1).long()

            new_data = Data(
                id=data.id,
                idx=data.idx,
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                y=y
            )

            new_datalist.append(new_data)

        return Batch.from_data_list(new_datalist)


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
    model = MyOutRGAT().to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_function = torch.nn.BCELoss().to(device)

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
        for i, data in enumerate(train_loader):
            model.zero_grad()

            data = data.to(device)
            y = data.y.float()

            h, y_hat = model(data)
            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 50 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

        # 验证
        total_val_loss = 0.0
        y_hat_total = torch.randn(0, 5)
        y_total = torch.randn(0, 5)

        xs = torch.randn(0, 128)
        ys = []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):

                data = data.to(device)
                y = data.y.float()

                h, y_hat = model(data)
                loss = loss_function(y_hat, y)

                # 用来计算整体指标
                total_val_loss += loss.item()
                y_hat = transact(y_hat).to(device)
                y_hat_total = torch.cat([y_hat_total.cpu(), OR2OEN(y_hat)], dim=0)
                y_total = torch.cat([y_total.cpu(), OR2OEN(y)], dim=0)

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
        auc = roc_auc_score(y_total.cpu(), y_hat_total.cpu())
        aod = AOD(y_total.cpu(), y_hat_total.cpu())
        print(f"验证集 accuracy_score: {float_to_percent(acc)}")
        print(f"验证集 auc: {float_to_percent(auc)}")
        print(f"验证集 aod: {float_to_percent(aod)}")

        if acc > best_acc:
            print(f"***当前模型的平衡准确率表现最好，被记为表现最好的模型***\n")
            best_model = model
            best_acc = acc

        print(f"***保存tsne中***\n")

        xs = xs.numpy()
        ys = np.array(ys)

        visual(xs, ys, epoch + 1)

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 测试集

    correct = {}
    wrong = {}

    total_val_loss = 0.0
    y_hat_total = torch.randn(0, 5)
    y_total = torch.randn(0, 5)

    xs = torch.randn(0, 128)
    ys = []

    record_file = open(os.path.join('./', 'result', 'result.txt'), 'w')

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            data = data.to(device)
            y = data.y.float()
            ids = data.id

            h, y_hat = model(data)
            loss = loss_function(y_hat, y)

            # 用来计算整体指标
            total_val_loss += loss.item()
            y_hat = transact(y_hat).to(device)
            y_hat_total = torch.cat([y_hat_total.cpu(), OR2OEN(y_hat)], dim=0)
            y_total = torch.cat([y_total.cpu(), OR2OEN(y)], dim=0)

            # 这里帮助可视化
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

            # 我们记录所有预测错误的语句
            for j in range(size):
                fac = y.cpu()[j]
                pre = y_hat.cpu()[j]
                id = ids[j]

                if torch.equal(fac, pre):
                    correct[id] = tensor2label(fac)
                else:
                    wrong[id] = 'expecting ' + tensor2label(fac) + ' but got ' + tensor2label(pre)

    print(f"测试集整体Loss: {total_val_loss}")

    acc = accuracy_score(y_total.cpu(), y_hat_total.cpu())
    auc = roc_auc_score(y_total.cpu(), y_hat_total.cpu())
    class_acc = class_acc(y_total.cpu(), y_hat_total.cpu())
    aod = AOD(y_total.cpu(), y_hat_total.cpu())
    print(f"测试集 accuracy_score: {float_to_percent(acc)}")
    print(f"测试集 auc: {float_to_percent(auc)}")
    print(f"测试集 aod: {float_to_percent(aod)}")
    print(f"测试集 各类别准确率: {class_acc}")

    print(f"***保存tsne中***\n")

    xs = xs.numpy()
    ys = np.array(ys)

    visual(xs, ys, -1)

    print(f"***写入测试集测试结果中***\n")
    record_file.write('预测正确的:\n')
    num = 0
    for key, value in correct.items():
        record_file.write(f"    -{num}. {key} : {value}\n")
        num += 1

    record_file.write('——————————————————————————————————————————————————————\n')

    record_file.write('预测错误的:\n')
    num = 0
    for key, value in wrong.items():
        record_file.write(f"    -{num}. {key} : {value}\n")
        num += 1

    record_file.close()
