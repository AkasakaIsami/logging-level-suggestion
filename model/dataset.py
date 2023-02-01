import configparser
import os
import time
from functools import cmp_to_key

import pandas as pd
import pydot
import torch
from gensim.models import Word2Vec
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from typing import Tuple

from util import cut_word


class MyDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, project=None, methods=None):
        self.word2vec = None
        self.embeddings = None
        self.project = project
        self.methods = methods
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(MyDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        paths = ['']
        return paths

    @property
    def processed_file_names(self):
        return ['', 'method_info.pkl', 'dataset.pt']

    def process(self):
        print("数据没有被预处理过，先进行预处理。")
        start = time.time()

        # 先读取一些配置
        cf = configparser.ConfigParser()
        cf.read('config.ini')
        ratio = cf.get('data', 'ratio')
        embedding_dim = cf.getint('embedding', 'dim')

        if not os.path.exists(self.processed_paths[0]):
            os.makedirs(self.processed_paths[0])
        record_file_path = os.path.join(self.processed_paths[0], 'dataset_info.txt')
        record_file = open(record_file_path, 'w')

        # 先导入词嵌入矩阵
        project_root = self.raw_paths[0]
        word2vec_path = os.path.join(project_root, self.project + '_w2v_' + str(embedding_dim) + '.model')
        word2vec = Word2Vec.load(word2vec_path).wv
        embeddings = torch.from_numpy(word2vec.vectors)
        embeddings = torch.cat([embeddings, torch.zeros(1, embedding_dim)], dim=0)
        self.embeddings = embeddings
        self.word2vec = word2vec

        # 开始遍历所有函数
        datalist = []
        method_info = pd.DataFrame(columns=['id', 'ASTs', 'edges'])

        # 计数器
        level_count = {
            'trace': 0,
            'debug': 0,
            'info': 0,
            'warn': 0,
            'error': 0
        }

        for index, item in tqdm(self.methods.iterrows(), total=len(self.methods)):
            clz = item['class']

            method = item['method']
            path = os.path.join(project_root, clz, method)

            files = os.listdir(path)
            method_graph_file = None
            statement_graphs_file = None
            for file in files:
                if file == '.DS_Store':
                    continue
                elif file.startswith('statements'):
                    statement_graphs_file = file
                else:
                    method_graph_file = file

            # 开始解析函数图
            method_graph_path = os.path.join(path, method_graph_file)
            method_graphs = pydot.graph_from_dot_file(method_graph_path)
            method_graph = method_graphs[0]

            y, cfg_edge_index, dfg_edge_index, lines, log_idx, method_count = self.process_method_dot(method_graph)

            level_count['trace'] += method_count['trace']
            level_count['debug'] += method_count['debug']
            level_count['info'] += method_count['info']
            level_count['warn'] += method_count['warn']
            level_count['error'] += method_count['error']

            # 解析所有语句图
            statements_path = os.path.join(path, statement_graphs_file)
            statement_graphs = pydot.graph_from_dot_file(statements_path)

            # 简单做个验证
            if len(statement_graphs) != len(y):
                print(f"!!!!!!!!!!!!!!!!!!{clz}的{method}解析的有问题！！！")

            asts = []
            num_statements = len(statement_graphs)
            id = clz + '_' + method
            for i in range(num_statements):
                statement_graph = statement_graphs[i]

                # Step1: 先构建存在method_info.pkl里的函数信息数据
                is_log = log_idx[i] == 1
                ast_x, ast_edge_index, msg_embd = self.process_statement_dot(graph=statement_graph, weight=None,
                                                                             is_log=is_log)
                ast_data = Data(
                    x=ast_x,
                    edge_index=ast_edge_index,
                )
                asts.append(ast_data)

                # Step2: 再构建statement数据集
                if is_log:
                    # 当前处理的ast是日志语句
                    index = torch.zeros(num_statements, 1)
                    index[i] = 1
                    statement_data = Data(
                        id=id,
                        line=lines[i],
                        idx=index,
                        y=torch.tensor(y[i]),
                        msg=msg_embd
                    )

                    datalist.append(statement_data)
            method_info.loc[len(method_info)] = [id, asts, (cfg_edge_index, dfg_edge_index)]

        # 数据都在内存里了 开始保存
        end = time.time()
        print(f"全部数据读取至内存完毕，开始以{ratio}切分数据集")

        print(
            f"数据集切分完毕，日志语句{len(datalist)}条，函数总量{len(method_info)}")

        print("现在开始对数据进行持久化存储……")
        method_info.to_pickle(self.processed_paths[1])

        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[2])

        '''
        每次构建数据集时，返回当前数据集的信息。信息包括：
            -  数据总量
            -  正样本量
            -  负样本量
            -  总函数量
            -  耗时
            还有训练集、验证集、测试集的详细信息。信息包括：
            -   集合数据总量
            -   集合数据正样本量
            -   集合数据负样本量
        '''

        record_file.write(f"数据集构建完成，下面是一些数据集相关信息：\n")
        record_file.write(f"    - 目标项目：{self.project}\n")
        record_file.write(f"    - 数据总量：{len(datalist)}\n")
        record_file.write(f"    - Trace量：{level_count['trace']}\n")
        record_file.write(f"    - Debug量：{level_count['debug']}\n")
        record_file.write(f"    - Info量：{level_count['info']}\n")
        record_file.write(f"    - Warn量：{level_count['warn']}\n")
        record_file.write(f"    - Error量：{level_count['error']}\n")
        record_file.write(f"    - 总耗时：{end - start}秒\n")

    def process_method_dot(self, graph) -> Tuple[list, torch.Tensor, torch.Tensor, list, list, dict]:
        """
        处理函数的dot，返回当前函数的图结构
        返回值：y, cfg_edge_index, dfg_edge_index, lines, log_idx
        """
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]

        # 对nodes进行自定义排序
        def cmp(x, y):
            a = int(x.get_name()[1:])
            b = int(y.get_name()[1:])

            return 1 if a > b else -1 if a < b else 0

        nodes.sort(key=cmp_to_key(cmp))
        node_num = len(nodes)

        levels = ['trace', 'debug', 'info', 'warn', 'error']
        level_dict = {
            'default': [0, 0, 0, 0, 0],
            levels[0]: [1, 0, 0, 0, 0],
            levels[1]: [1, 1, 0, 0, 0],
            levels[2]: [1, 1, 1, 0, 0],
            levels[3]: [1, 1, 1, 1, 0],
            levels[4]: [1, 1, 1, 1, 1]
        }

        level_count = {
            levels[0]: 0,
            levels[1]: 0,
            levels[2]: 0,
            levels[3]: 0,
            levels[4]: 0
        }

        y = []
        # 存了每个语句的行数 数量和节点数量对应
        lines = []
        log_idx = []

        index_map = {}
        for i in range(node_num):
            node = nodes[i]
            line = node.get_attributes()['line']
            lines.append(int(line))

            node_index = int(node.get_name()[1:])
            index_map[node_index] = i

            # 判断是不是日志语句
            isLogStmt = '"true"' in node.get_attributes()['isLogStmt']

            if isLogStmt:
                level = node.get_attributes()['level']
                level = level[1:len(level) - 1]

                # TODO: 暂时无法解析的日志（Lambda语句的case）算作正常语句！
                if level in levels:
                    level_count[level] += 1
                    log_idx.append(1)
                    y.append(level_dict[level])
                else:
                    log_idx.append(0)
                    y.append(level_dict['default'])
            else:
                log_idx.append(0)
                y.append(level_dict['default'])

        edges = graph.get_edge_list()
        edge_0_cfg = []
        edge_1_cfg = []
        edge_0_dfg = []
        edge_1_dfg = []

        for edge in edges:
            source = int(edge.get_source()[1:])
            destination = int(edge.get_destination()[1:])

            if source >= len(index_map) or destination >= len(index_map):
                continue

            source = index_map[source]
            destination = index_map[destination]

            color = edge.get_attributes()['color']

            if color == 'red':
                edge_0_cfg.append(source)
                edge_1_cfg.append(destination)
            elif color == 'green':
                edge_0_dfg.append(source)
                edge_1_dfg.append(destination)

        edge_0_cfg = torch.as_tensor(edge_0_cfg)
        edge_1_cfg = torch.as_tensor(edge_1_cfg)
        edge_0_cfg = edge_0_cfg.reshape(1, len(edge_0_cfg))
        edge_1_cfg = edge_1_cfg.reshape(1, len(edge_1_cfg))

        edge_0_dfg = torch.as_tensor(edge_0_dfg)
        edge_1_dfg = torch.as_tensor(edge_1_dfg)
        edge_0_dfg = edge_0_dfg.reshape(1, len(edge_0_dfg))
        edge_1_dfg = edge_1_dfg.reshape(1, len(edge_1_dfg))

        cfg_edge_index = torch.cat([edge_0_cfg, edge_1_cfg], dim=0)
        dfg_edge_index = torch.cat([edge_0_dfg, edge_1_dfg], dim=0)

        return y, cfg_edge_index, dfg_edge_index, lines, log_idx, level_count

    def process_statement_dot(self, graph, weight, is_log):
        """
        这个函数返回ST-AST的特征矩阵和邻接矩阵
        特征矩阵需要根据语料库构建……

        :param is_log:
        :param weight:
        :param graph: ST-AST
        :return: 特征矩阵和邻接矩阵
        """

        def word_to_vec(token):
            """
            词转词嵌入
            :param token:
            :return: 返回一个代表词嵌入的ndarray
            """
            max_token = self.word2vec.vectors.shape[0]
            index = [self.word2vec.key_to_index[token] if token in self.word2vec.key_to_index else max_token]
            return self.embeddings[index]

        def tokens_to_embedding(tokens, weight):
            """
            对于多token组合的节点 可以有多种加权求和方式
            这里简单的求平均先

            :param tokens:节点的token序列
            :return: 最终的节点向量
            """
            result = torch.zeros([1, 128], dtype=torch.float)

            for token in tokens:
                token_embedding = word_to_vec(token)
                if weight is not None:
                    token_weight = weight[token] if weight.has_key(token) else 0
                    token_embedding = token_embedding * token_weight
                result = result + token_embedding

            count = len(tokens)
            result = result / count
            return result

        x = []
        msgs = []
        nodes = graph.get_node_list()
        if len(graph.get_node_list()) > 0 and graph.get_node_list()[-1].get_name() == '"\\n"':
            nodes = graph.get_node_list()[:-1]

        for node in nodes:
            node_str = node.get_attributes()['label']
            # token 可能是多种形势，要先切分
            tokens = cut_word(node_str)
            # 多token可以考虑不同的合并方式
            node_embedding = tokens_to_embedding(tokens, weight)
            x.append(node_embedding)
            if is_log and node_str.startswith('\"value='):
                msgs.append(node_embedding)

        x = torch.cat(x)
        msg_embd = torch.zeros(1, 128)
        for msg in msgs:
            msg_embd += msg
        msg_embd /= len(msgs)

        edges = graph.get_edge_list()
        edge_0 = []
        edge_1 = []

        for edge in edges:
            source = int(edge.get_source()[1:])
            destination = int(edge.get_destination()[1:])
            edge_0.append(source)
            edge_1.append(destination)

        edge_0 = torch.as_tensor(edge_0, dtype=torch.int)
        edge_1 = torch.as_tensor(edge_1, dtype=torch.int)
        edge_0 = edge_0.reshape(1, len(edge_0))
        edge_1 = edge_1.reshape(1, len(edge_1))

        edge_index = torch.cat([edge_0, edge_1], dim=0)

        return x, edge_index, msg_embd

    def get_labels(self, ):
        return self.data.y
