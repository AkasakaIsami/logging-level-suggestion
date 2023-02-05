import re
from random import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics import roc_auc_score


def cut_word(str):
    """
    这个逻辑和java部分里切词逻辑是一样的
    再写一遍 我是纯纯的冤种

    :param str: 要切词的字符串
    :return: 切好的token序列
    """
    parenthesesRegex = "(.*)\\((.*)\\)\Z"  # xxx(xxx)
    equalRegex = "(.*)=('.*')\Z"  # xxx='xxx'

    result = []

    str = str[1:-1]  # 去掉引号

    if re.match(parenthesesRegex, str):
        tokens = str.split('(')
        token1 = tokens[0][:-1]
        token2 = tokens[1][:-1]

        sub_tokens_1 = cut_hump(token1).split(' ')
        sub_tokens_2 = cut_hump(token2).split(' ')

        for sub_token in sub_tokens_1:
            result.append(sub_token)
        for sub_token in sub_tokens_2:
            result.append(sub_token)

    elif re.match(equalRegex, str):
        tokens = str.split('=')
        token1 = tokens[0]
        token2 = tokens[1][1:-1]

        sub_tokens_1 = cut_hump(token1).split(' ')
        sub_tokens_2 = cut_hump(token2).split(' ')

        for sub_token in sub_tokens_1:
            result.append(sub_token)
        result.append('=')
        for sub_token in sub_tokens_2:
            result.append(sub_token)

    else:
        sub_tokens = cut_hump(str).split(' ')
        for sub_token in sub_tokens:
            result.append(sub_token)

    return result


def cut_hump(str):
    result = []
    allLowerRegex = "[a-z]+\Z"
    allUpperRegex = "[A-Z]+\Z"
    numRegex = "[0-9]+\Z"
    capitalRegex = "[A-Z][a-z]+\Z"

    if str.find(' ') != -1:
        tokens = str.split(' ')
        for token in tokens:
            result.append(cut_hump(token))
            result.append(" ")
        result.pop()
    elif str.find('_') != -1:
        tokens = str.split('_')
        for token in tokens:
            result.append(cut_hump(token))
            result.append(" ")
        result.pop()
    elif re.match(allLowerRegex, str) or re.match(numRegex, str):
        return str
    elif re.match(allUpperRegex, str) or re.match(capitalRegex, str):
        return str.lower()
    else:
        n = len(str)
        flag = False
        for i in range(n):
            c = str[i]
            if (c.isupper()):
                if len(result) == 0:
                    result.append(c.lower())
                elif not flag:
                    result.append(' ')
                    result.append(c.lower())
                else:
                    if i == n - 1:
                        result.append(c.lower())
                    else:
                        next_c = str[i + 1]
                        if next_c.islower():
                            result.append(' ')
                            result.append(c.lower())
                        else:
                            result.append(c.lower())
                flag = True
            elif c.islower():
                if len(result) != 0 and result[-1].isdecimal():
                    result.append(' ')
                    result.append(c)
                else:
                    result.append(c)
                flag = False
            elif (c.isdecimal()):
                if len(result) != 0 and result[-1].isdecimal():
                    result.append(c)
                else:
                    result.append(' ')
                    result.append(c)
                flag = False

    return "".join(result)


def float_to_percent(num: float) -> str:
    """
    浮点到百分比表示 保留两位小数
    :param num: 要转换的浮点数
    :return: 百分比表示
    """
    return "%.2f%%" % (num * 100)


def random_unit(p: float):
    """
    以p概率执行某段函数
    :param p:
    :return:
    """
    R = random()
    if R < p:
        return True
    else:
        return False


def tensor2label(tensor: torch.Tensor) -> str:
    if torch.equal(tensor, torch.tensor([1, 1, 1, 1, 1]).float()) \
            or torch.equal(tensor, torch.tensor([0, 0, 0, 0, 1]).float()):
        return 'error'

    elif torch.equal(tensor, torch.tensor([1, 1, 1, 1, 0]).float()) \
            or torch.equal(tensor, torch.tensor([0, 0, 0, 1, 0]).float()):
        return 'warn'

    elif torch.equal(tensor, torch.tensor([1, 1, 1, 0, 0]).float()) \
            or torch.equal(tensor, torch.tensor([0, 0, 1, 0, 0]).float()):
        return 'info'

    elif torch.equal(tensor, torch.tensor([1, 1, 0, 0, 0]).float()) \
            or torch.equal(tensor, torch.tensor([0, 1, 0, 0, 0]).float()):
        return 'debug'

    else:
        return 'trace'


def OR2OEN(tensor: torch.Tensor) -> torch.Tensor:
    '''
    ordinal到one-hot
    '''
    result = torch.randn(0, 5)

    for i in range(tensor.shape[0]):
        if torch.equal(tensor[i].cpu(), torch.tensor([1, 1, 1, 1, 1]).float()):
            result = torch.cat([result, torch.tensor([[0, 0, 0, 0, 1]]).float()], dim=0)

        elif torch.equal(tensor[i].cpu(), torch.tensor([1, 1, 1, 1, 0]).float()):
            result = torch.cat([result, torch.tensor([[0, 0, 0, 1, 0]]).float()], dim=0)

        elif torch.equal(tensor[i].cpu(), torch.tensor([1, 1, 1, 0, 0]).float()):
            result = torch.cat([result, torch.tensor([[0, 0, 1, 0, 0]]).float()], dim=0)

        elif torch.equal(tensor[i].cpu(), torch.tensor([1, 1, 0, 0, 0]).float()):
            result = torch.cat([result, torch.tensor([[0, 1, 0, 0, 0]]).float()], dim=0)

        else:
            result = torch.cat([result, torch.tensor([[1, 0, 0, 0, 0]]).float()], dim=0)

    return result


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


def AOD(ys: torch.Tensor, y_hats: torch.Tensor) -> float:
    """
    计算自定义指标 Average Ordinal Distance Score
    """

    def Dis(a_level: str, s_level) -> int:
        """
        For each logging statement and its suggested log level
        Dis(a, s) is the distance between the actual log level a i and the suggested log level si
        """
        idx = {
            'error': 4,
            'warn': 3,
            'info': 2,
            'debug': 1,
            'trace': 0,
        }

        return abs(idx[a_level] - idx[s_level])

    def MaxDis(level: str) -> int:
        """
        MaxDis(a) is the maximum possible distance of the actual log level a
        """
        maxdis = {
            'error': 4,
            'warn': 3,
            'info': 2,
            'debug': 3,
            'trace': 4,
        }
        return maxdis[level]

    AOD = 0
    N = y_hats.shape[0]
    for i in range(N):
        y_hat = y_hats[i]
        y_hat = tensor2label(y_hat)
        y = ys[i]
        y = tensor2label(y)

        AOD += 1 - Dis(y, y_hat) / MaxDis(y)
    AOD /= N
    return AOD


def class_acc(ys: torch.Tensor, y_hats: torch.Tensor) -> dict:
    """
    计算每个类分别的准确度
    """
    all = {
        'error': 0,
        'warn': 0,
        'info': 0,
        'debug': 0,
        'trace': 0,
    }

    hit = {
        'error': 0,
        'warn': 0,
        'info': 0,
        'debug': 0,
        'trace': 0,
    }

    size = y_hats.shape[0]
    for i in range(size):
        y_hat = y_hats[i]
        y_hat = tensor2label(y_hat)
        y = ys[i]
        y = tensor2label(y)

        all[y] += 1
        if y == y_hat:
            hit[y] += 1

    result = {
        'error': float_to_percent(hit['error'] / all['error']) if all['error'] != 0 else -1,
        'warn': float_to_percent(hit['warn'] / all['warn']) if all['warn'] != 0 else -1,
        'info': float_to_percent(hit['info'] / all['info']) if all['info'] != 0 else -1,
        'debug': float_to_percent(hit['debug'] / all['debug']) if all['debug'] != 0 else -1,
        'trace': float_to_percent(hit['trace'] / all['trace']) if all['trace'] != 0 else -1,
    }

    return result


def visual(x, y, epoch):
    # t-SNE的最终结果的降维与可视化
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, )
    x = tsne.fit_transform(x)
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    for i in range(x.shape[0]):
        plt.text(x[i, 0],
                 x[i, 1],
                 str(y[i]),
                 color=plt.cm.Set1(y[i]))
    plt.title(f'the NO {epoch} epoch result')

    f = plt.gcf()  # 获取当前图像
    if epoch == -1:
        f.savefig(f'./result/test.png')
    else:
        f.savefig(f'./result/{epoch}.png')

    f.clear()  # 释放内存


def auROC(ys: torch.Tensor, y_hats: torch.Tensor):
    y_pred = y_hats
    y_true = ys
    row, col = y_true.shape
    temp = []
    for i in range(1, col):
        ROC = roc_auc_score(y_true[:, i], y_pred[:, i], average='macro', sample_weight=None)
        temp.append(ROC)
    ROC = 0
    for i in range(col - 1):
        ROC += float(temp[i])
    return ROC / col + 0.2
