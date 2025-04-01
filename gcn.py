import numpy as np
import scipy as sp
import torch
from torch import nn
import dgl
import torch.nn.functional as F
import scipy.sparse as sp
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.__version__)
print(torch.version.cuda)

torch.cuda.empty_cache()


def convert_sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.FloatTensor(coo.data)
    # from a sparse matrix to a sparse float tensor
    sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
    return sp_tensor


def create_adjacency_matrices(G, num_mirna, num_disease):
    # 获取边的起始节点和目标节点
    edges_src, edges_dst = G.edges()

    # 初始化 miRNA -> disease 和 disease -> miRNA 的邻接矩阵
    mirna_R = torch.zeros((num_mirna, num_disease), dtype=torch.float32)
    disease_R = torch.zeros((num_disease, num_mirna), dtype=torch.float32)

    # 将 edges_src 和 edges_dst 中的边信息对应到 miRNA 和 disease 的节点范围
    for src, dst in zip(edges_src, edges_dst):
        if src < num_disease <= dst:  # disease -> miRNA
            disease_R[src, dst - num_disease] = 1
        elif src >= num_disease > dst:  # miRNA -> disease
            mirna_R[src - num_disease, dst] = 1

    # 计算 miRNA -> disease 和 disease -> miRNA 的度矩阵
    mirna_degree = mirna_R.sum(dim=1)  # miRNA 的度（每行的和）
    disease_degree = disease_R.sum(dim=1)  # disease 的度（每行的和）

    # 避免度为零的情况，防止除零错误
    mirna_degree[mirna_degree == 0] = 1
    disease_degree[disease_degree == 0] = 1

    # 计算 D^(-1/2) * A * D^(-1/2) 归一化邻接矩阵
    mirna_D_inv_sqrt = torch.diag(1.0 / torch.sqrt(mirna_degree))
    disease_D_inv_sqrt = torch.diag(1.0 / torch.sqrt(disease_degree))

    # 归一化邻接矩阵
    mirna_R = mirna_D_inv_sqrt @ mirna_R @ disease_D_inv_sqrt
    disease_R = disease_D_inv_sqrt @ disease_R @ mirna_D_inv_sqrt

    return mirna_R, disease_R


class NIE_GCN(nn.Module):
    def __init__(self, G, train_mirna, train_disease, dim, GCNLayer, beta, dropout, device):
        super(NIE_GCN, self).__init__()
        self.G = G.to(device)
        self.train_mirna = train_mirna
        self.train_disease = train_disease
        self.dim = dim
        self.GCNLayer = GCNLayer
        self.beta = beta
        self.device = device
        self.count = 10
        self.showtime = 0

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.num_mirna = 495
        self.num_disease = 383

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], self.dim, bias=False)  # 特征变换
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], self.dim, bias=False)  # 特征变换
        self.dropout = nn.Dropout(dropout)

        # 初始化邻接矩阵
        mirna_R, disease_R = create_adjacency_matrices(self.G, self.num_mirna, self.num_disease)

        # 将邻接矩阵转换为稀疏张量
        indices_m = torch.nonzero(mirna_R, as_tuple=True)  # 获取非零元素的行和列索引
        values_m = mirna_R[indices_m].float()  # 提取这些索引处的值并转换为浮点数
        torch.sparse.FloatTensor(
            torch.stack(indices_m),  # 将行和列索引堆叠在一起形成 2 x N 的矩阵
            values_m,  # 对应的非零值
            mirna_R.size()  # 原始矩阵的大小
        ).to(self.device)

        indices_d = torch.nonzero(disease_R, as_tuple=True)  # 获取非零元素的行和列索引
        values_d = disease_R[indices_d].float()  # 提取这些索引处的值并转换为浮点数
        torch.sparse.FloatTensor(
            torch.stack(indices_d),  # 将行和列索引堆叠在一起形成 2 x N 的矩阵
            values_d,  # 对应的非零值
            disease_R.size()  # 原始矩阵的大小
        ).to(self.device)

        self.reset_parameters()

        # 定义一个全连接的三层神经网络
        self.attention_dense = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, 1, bias=False)
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()
        self.activation_layer = nn.Tanh()
        self.attention_activation = nn.ReLU()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

    def update_attention_A_mi(self, G, train_mirna, train_disease):
        # self.mirna_embedding = torch.sparse.mm(self.miRNA_R, self.disease_embedding.weight)
        train_mirna = torch.tensor(train_mirna) - 383
        train_disease = torch.tensor(train_disease)

        fold_len = len(train_mirna) // self.count  # 计算训练集中的用户列表 train_user 的分块长度 fold_len
        attention_score = []

        for i in range(self.count):
            start = i * fold_len  # 计算当前分块的起始位置 start
            if i == self.count - 1:
                end = len(train_mirna)  # 如果这是最后一个分块，设置 end 为训练集用户数据的最后一个位置
            else:
                end = (i + 1) * fold_len  # 如果不是最后一个分块，设置 end 为下一个分块的起始位置
            A_score = self.attention_score(G, train_mirna[start:end], train_disease[start:end])
            # 调用 attention_score 函数，计算当前分块中用户和物品之间的注意力得分。传入的是用户和物品的训练数据。
            attention_score.append(A_score)  # 添加到列表里

        attention_score = torch.cat(attention_score).squeeze()  # 将所有分块的注意力得分连接成一个完整的张量，并使用 squeeze() 函数去掉不必要的维度
        new_attention_score = attention_score.detach().cpu()  # 使用 detach() 将注意力得分从计算图中分离出来
        new_attention_score = np.exp(new_attention_score)  # 对注意力得分进行指数运算，将得分转化为非负值，以便于后续归一化处理

        train_mirna = train_mirna.cpu()
        train_disease = train_disease.cpu()

        new_R = sp.coo_matrix((new_attention_score, (train_mirna, train_disease)),
                              # 使用稀疏矩阵 coo_matrix 将新的注意力得分与用户-物品对关联起来
                              shape=(self.num_mirna, self.num_disease))  # 构建新的用户-物品关联矩阵 new_R形状为.

        new_R = new_R / np.power(new_R.sum(axis=1), self.beta)  # 对新的注意力矩阵进行归一化处理
        new_R[np.isinf(new_R)] = 0.  # 处理归一化过程中出现的无穷大值，将其设置为 0，避免异常值干扰计算
        new_R = sp.csr_matrix(new_R)  # 将矩阵 new_R 转换为更高效的稀疏矩阵格式 csr_matrix（压缩稀疏行格式）
        mirna_R = convert_sp_mat_to_sp_tensor(new_R).coalesce().to(self.device)
        return mirna_R

    def update_attention_A_di(self, G, train_mirna, train_disease):
        # self.disease_embedding = torch.sparse.mm(self.siease_R, self.mirna_embedding.weight)
        train_mirna = torch.tensor(train_mirna) - 383
        train_disease = torch.tensor(train_disease)

        fold_len_d = len(train_disease) // self.count  # 计算训练集中的用户列表 train_user 的分块长度 fold_len
        attention_score_d = []

        for i in range(self.count):
            start = i * fold_len_d  # 计算当前分块的起始位置 start
            if i == self.count - 1:
                end = len(train_disease)  # 如果这是最后一个分块，设置 end 为训练集用户数据的最后一个位置
            else:
                end = (i + 1) * fold_len_d  # 如果不是最后一个分块，设置 end 为下一个分块的起始位置
            A_score_d = self.attention_score(G, train_mirna[start:end], train_disease[start:end])
            # 调用 attention_score 函数，计算当前分块中用户和物品之间的注意力得分。传入的是用户和物品的训练数据。
            attention_score_d.append(A_score_d)  # 添加到列表里

        attention_score_d = torch.cat(attention_score_d).squeeze()  # 将所有分块的注意力得分连接成一个完整的张量，并使用 squeeze() 函数去掉不必要的维度
        new_attention_score_d = attention_score_d.detach().cpu()  # 使用 detach() 将注意力得分从计算图中分离出来
        new_attention_score_d = np.exp(new_attention_score_d)  # 对注意力得分进行指数运算，将得分转化为非负值，以便于后续归一化处理

        train_mirna = train_mirna.cpu()
        train_disease = train_disease.cpu()

        new_R_d = sp.coo_matrix((new_attention_score_d, (train_disease, train_mirna)),
                                # 使用稀疏矩阵 coo_matrix 将新的注意力得分与用户-物品对关联起来
                                shape=(self.num_disease, self.num_mirna))  # 构建新的用户-物品关联矩阵 new_R形状为.

        new_R_d = new_R_d / np.power(new_R_d.sum(axis=1), self.beta)  # 对新的注意力矩阵进行归一化处理
        new_R_d[np.isinf(new_R_d)] = 0.  # 处理归一化过程中出现的无穷大值，将其设置为 0，避免异常值干扰计算
        new_R_d = sp.csr_matrix(new_R_d)  # 将矩阵 new_R 转换为更高效的稀疏矩阵格式 csr_matrix（压缩稀疏行格式）

        disease_R = convert_sp_mat_to_sp_tensor(new_R_d).coalesce().to(self.device)
        return disease_R

    def attention_score(self, G, mirna, disease):
        mirna_embedding = self.dropout(self.m_fc(G.ndata['m_sim'][-495:, :].to(self.device)))
        disease_embedding = self.dropout(self.d_fc(G.ndata['d_sim'][:383, :].to(self.device)))
        assert len(mirna) == len(disease)  # 确保传入的用户和物品数量相等。
        mirna = torch.tensor(mirna).to(self.device)  # 将用户列表转换为张量
        # mirna = torch.tensor(mirna, dtype=torch.float32).to(self.device)
        disease = torch.tensor(disease, dtype=torch.float32).to(self.device)
        # disease = torch.tensor(disease).to(self.device)
        mirna_embed = mirna_embedding[mirna.long()]  # 根据用户索引获取对应的用户嵌入。
        disease_embed = disease_embedding[disease.long()]
        embedding = nn.functional.relu(
            self.attention_dropout(torch.cat([mirna_embed, disease_embed], dim=1))
        )  # 将用户和物品的嵌入拼接在一起，并通过 ReLU 激活函数进行非线性变换
        score = self.attention_dense(embedding)  # 使用全连接层（attention_dense）计算用户和物品之间的注意力得分

        return score.squeeze()  # 返回计算出的注意力得分，并去掉多余的维度。

    # 该函数实现了图卷积聚合操作，通过若干层的稀疏矩阵乘法，逐步更新两个节点的嵌入表示
    def forward(self, G, train_mirna, train_disease):
        mirna_embedding = self.dropout(self.m_fc(G.ndata['m_sim'][-495:, :].to(self.device)))
        disease_embedding = self.dropout(self.d_fc(G.ndata['d_sim'][:383, :].to(self.device)))
        all_mirna_embeddings = []
        all_disease_embeddings = []  # 初始化两个列表
        mirna_R = self.update_attention_A_mi(G, train_mirna, train_disease).to_dense()
        disease_R = self.update_attention_A_di(G, train_mirna, train_disease).to_dense()
        # aggregate
        for layer in range(self.GCNLayer):
            mirna_embedding = self.activation_layer(torch.mm(mirna_R, disease_embedding))
            # 使用稀疏矩阵乘法（torch.sparse.mm）将用户-物品关系矩阵self.user_R与物品嵌入item_embedding相乘
            disease_embedding = self.activation_layer(torch.mm(disease_R, mirna_embedding))

            all_mirna_embeddings.append(mirna_embedding)
            all_disease_embeddings.append(disease_embedding)

            final_mirna_embeddings = torch.stack(all_mirna_embeddings, dim=1)
            final_mirna_embeddings = torch.sum(final_mirna_embeddings, dim=1)

            final_disease_embeddings = torch.stack(all_disease_embeddings, dim=1)
            final_disease_embeddings = torch.sum(final_disease_embeddings, dim=1)

        return final_mirna_embeddings, final_disease_embeddings


class NIE_GCN_Metasubgraph(nn.Module):
    def __init__(self, G, train_mirna, train_disease, dim, GCNLayer, beta, dropout, device):
        super(NIE_GCN_Metasubgraph, self).__init__()
        self.G = G.to(device)
        self.train_mirna = train_mirna
        self.train_disease = train_disease
        self.dim = dim
        self.GCNLayer = GCNLayer
        self.beta = beta
        self.device = device
        self.count = 10
        self.showtime = 0

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.num_mirna = 495
        self.num_disease = 383

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], self.dim, bias=False)  # 特征变换
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], self.dim, bias=False)  # 特征变换
        self.dropout = nn.Dropout(dropout)

        # 初始化邻接矩阵
        mirna_R, disease_R = create_adjacency_matrices(self.G, self.num_mirna, self.num_disease)

        # 将邻接矩阵转换为稀疏张量
        indices_m = torch.nonzero(mirna_R, as_tuple=True)  # 获取非零元素的行和列索引
        values_m = mirna_R[indices_m].float()  # 提取这些索引处的值并转换为浮点数
        torch.sparse.FloatTensor(
            torch.stack(indices_m),  # 将行和列索引堆叠在一起形成 2 x N 的矩阵
            values_m,  # 对应的非零值
            mirna_R.size()  # 原始矩阵的大小
        ).to(self.device)

        indices_d = torch.nonzero(disease_R, as_tuple=True)  # 获取非零元素的行和列索引
        values_d = disease_R[indices_d].float()  # 提取这些索引处的值并转换为浮点数
        torch.sparse.FloatTensor(
            torch.stack(indices_d),  # 将行和列索引堆叠在一起形成 2 x N 的矩阵
            values_d,  # 对应的非零值
            disease_R.size()  # 原始矩阵的大小
        ).to(self.device)

        self.reset_parameters()

        # 定义一个全连接的三层神经网络
        self.attention_dense = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, 1, bias=False)
        )

        self.activation = nn.Sigmoid()
        self.activation_layer = nn.Tanh()
        self.attention_activation = nn.ReLU()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

    def update_attention_A_mi(self, G, train_mirna, train_disease):
        # self.mirna_embedding = torch.sparse.mm(self.miRNA_R, self.disease_embedding.weight)
        train_mirna = torch.tensor(train_mirna) - 383
        train_disease = torch.tensor(train_disease)

        fold_len = len(train_mirna) // self.count  # 计算训练集中的用户列表 train_user 的分块长度 fold_len
        attention_score = []

        for i in range(self.count):
            start = i * fold_len  # 计算当前分块的起始位置 start
            if i == self.count - 1:
                end = len(train_mirna)  # 如果这是最后一个分块，设置 end 为训练集用户数据的最后一个位置
            else:
                end = (i + 1) * fold_len  # 如果不是最后一个分块，设置 end 为下一个分块的起始位置
            A_socre = self.attention_score(G, train_mirna[start:end], train_disease[start:end])
            # 调用 attention_score 函数，计算当前分块中用户和物品之间的注意力得分。传入的是用户和物品的训练数据。
            attention_score.append(A_socre)  # 添加到列表里

        attention_score = torch.cat(attention_score).squeeze()  # 将所有分块的注意力得分连接成一个完整的张量，并使用 squeeze() 函数去掉不必要的维度
        new_attention_score = attention_score.detach().cpu()  # 使用 detach() 将注意力得分从计算图中分离出来
        new_attention_score = np.exp(new_attention_score)  # 对注意力得分进行指数运算，将得分转化为非负值，以便于后续归一化处理

        train_mirna = train_mirna.cpu()
        train_disease = train_disease.cpu()

        new_R = sp.coo_matrix((new_attention_score, (train_mirna, train_disease)),
                              # 使用稀疏矩阵 coo_matrix 将新的注意力得分与用户-物品对关联起来
                              shape=(self.num_mirna, self.num_disease))  # 构建新的用户-物品关联矩阵 new_R形状为.

        # new_R = new_R / np.power(new_R.sum(axis=1), self.beta)  # 对新的注意力矩阵进行归一化处理
        row_sums = new_R.sum(axis=1)
        row_sums[row_sums == 0] = 1e-6  # 将和为零的行改为一个很小的数，避免除零
        new_R = new_R / np.power(row_sums, self.beta)
        new_R[np.isinf(new_R)] = 0.  # 处理归一化过程中出现的无穷大值，将其设置为 0，避免异常值干扰计算
        new_R = sp.csr_matrix(new_R)  # 将矩阵 new_R 转换为更高效的稀疏矩阵格式 csr_matrix（压缩稀疏行格式）
        mirna_R = convert_sp_mat_to_sp_tensor(new_R).coalesce().to(self.device)
        return mirna_R

    def update_attention_A_di(self, G, train_mirna, train_disease):
        # self.disease_embedding = torch.sparse.mm(self.siease_R, self.mirna_embedding.weight)
        train_mirna = torch.tensor(train_mirna) - 383
        train_disease = torch.tensor(train_disease)

        fold_len_d = len(train_disease) // self.count  # 计算训练集中的用户列表 train_user 的分块长度 fold_len
        attention_score_d = []

        for i in range(self.count):
            start = i * fold_len_d  # 计算当前分块的起始位置 start
            if i == self.count - 1:
                end = len(train_disease)  # 如果这是最后一个分块，设置 end 为训练集用户数据的最后一个位置
            else:
                end = (i + 1) * fold_len_d  # 如果不是最后一个分块，设置 end 为下一个分块的起始位置
            A_socre_d = self.attention_score(G, train_mirna[start:end], train_disease[start:end])
            # 调用 attention_score 函数，计算当前分块中用户和物品之间的注意力得分。传入的是用户和物品的训练数据。
            attention_score_d.append(A_socre_d)  # 添加到列表里

        attention_score_d = torch.cat(attention_score_d).squeeze()  # 将所有分块的注意力得分连接成一个完整的张量，并使用 squeeze() 函数去掉不必要的维度
        new_attention_score_d = attention_score_d.detach().cpu()  # 使用 detach() 将注意力得分从计算图中分离出来
        new_attention_score_d = np.exp(new_attention_score_d)  # 对注意力得分进行指数运算，将得分转化为非负值，以便于后续归一化处理

        train_mirna = train_mirna.cpu()
        train_disease = train_disease.cpu()

        new_R_d = sp.coo_matrix((new_attention_score_d, (train_disease, train_mirna)),
                                # 使用稀疏矩阵 coo_matrix 将新的注意力得分与用户-物品对关联起来
                                shape=(self.num_disease, self.num_mirna))  # 构建新的用户-物品关联矩阵 new_R形状为.

        # new_R_d = new_R_d / np.power(new_R_d.sum(axis=1), self.beta)  # 对新的注意力矩阵进行归一化处理
        row_sums = new_R_d.sum(axis=1)
        row_sums[row_sums == 0] = 1e-6  # 将和为零的行改为一个很小的数，避免除零
        new_R_d = new_R_d / np.power(row_sums, self.beta)
        new_R_d[np.isinf(new_R_d)] = 0.  # 处理归一化过程中出现的无穷大值，将其设置为 0，避免异常值干扰计算
        new_R_d = sp.csr_matrix(new_R_d)  # 将矩阵 new_R 转换为更高效的稀疏矩阵格式 csr_matrix（压缩稀疏行格式）

        disease_R = convert_sp_mat_to_sp_tensor(new_R_d).coalesce().to(self.device)
        return disease_R

    def attention_score(self, G, mirna, disease):
        mirna_embedding = self.dropout(self.m_fc(G.ndata['m_sim'][-495:, :].to(self.device)))
        disease_embedding = self.dropout(self.d_fc(G.ndata['d_sim'][:383, :].to(self.device)))
        assert len(mirna) == len(disease)  # 确保传入的用户和物品数量相等。
        mirna = torch.tensor(mirna).to(self.device)  # 将用户列表转换为张量
        # mirna = torch.tensor(mirna, dtype=torch.float32).to(self.device)
        disease = torch.tensor(disease, dtype=torch.float32).to(self.device)
        # disease = torch.tensor(disease).to(self.device)
        mirna_embed = mirna_embedding[mirna.long()]  # 根据用户索引获取对应的用户嵌入。
        disease_embed = disease_embedding[disease.long()]
        embedding = nn.functional.relu(
            torch.cat([mirna_embed, disease_embed], dim=1))  # 将用户和物品的嵌入拼接在一起，并通过 ReLU 激活函数进行非线性变换
        score = self.attention_dense(embedding)  # 使用全连接层（attention_dense）计算用户和物品之间的注意力得分

        return score.squeeze()  # 返回计算出的注意力得分，并去掉多余的维度。

    # 该函数实现了图卷积聚合操作，通过若干层的稀疏矩阵乘法，逐步更新两个节点的嵌入表示
    def forward(self, new_g, meta_path, train_mirna, train_disease):
        new_g = new_g.to(self.device)
        if meta_path == 'c':
            mirna_embedding = self.dropout(self.m_fc(new_g.ndata['m_sim'][-495:, :].to(self.device)))
            disease_embedding = self.dropout(self.d_fc(new_g.ndata['d_sim'][:383, :].to(self.device)))
            all_mirna_embeddings = []
            all_disease_embeddings = []  # 初始化两个列表
            mirna_R = self.update_attention_A_mi(new_g, train_mirna, train_disease).to_dense()
            disease_R = self.update_attention_A_di(new_g, train_mirna, train_disease).to_dense()
            # aggregate
            for layer in range(self.GCNLayer):
                mirna_embedding = self.activation_layer(torch.mm(mirna_R, disease_embedding))
                disease_embedding = self.activation_layer(torch.mm(disease_R, mirna_embedding))
                all_mirna_embeddings.append(mirna_embedding)
                all_disease_embeddings.append(disease_embedding)
                final_mirna_embeddings = torch.stack(all_mirna_embeddings, dim=1)
                final_mirna_embeddings_c = torch.sum(final_mirna_embeddings, dim=1)
                final_disease_embeddings = torch.stack(all_disease_embeddings, dim=1)
                final_disease_embeddings_c = torch.sum(final_disease_embeddings, dim=1)
            return final_mirna_embeddings_c, final_disease_embeddings_c

        if meta_path == 'e':
            mirna_embedding = self.dropout(self.m_fc(new_g.ndata['m_sim'][-495:, :].to(self.device)))
            disease_embedding = self.dropout(self.d_fc(new_g.ndata['d_sim'][:383, :].to(self.device)))
            all_mirna_embeddings = []
            all_disease_embeddings = []  # 初始化两个列表
            mirna_R = self.update_attention_A_mi(new_g, train_mirna, train_disease).to_dense()
            disease_R = self.update_attention_A_di(new_g, train_mirna, train_disease).to_dense()
            # aggregate
            for layer in range(self.GCNLayer):
                mirna_embedding = self.activation_layer(torch.mm(mirna_R, disease_embedding))
                disease_embedding = self.activation_layer(torch.mm(disease_R, mirna_embedding))
                all_mirna_embeddings.append(mirna_embedding)
                all_disease_embeddings.append(disease_embedding)
                final_mirna_embeddings = torch.stack(all_mirna_embeddings, dim=1)
                final_mirna_embeddings_e = torch.sum(final_mirna_embeddings, dim=1)
                final_disease_embeddings = torch.stack(all_disease_embeddings, dim=1)
                final_disease_embeddings_e = torch.sum(final_disease_embeddings, dim=1)
            return final_mirna_embeddings_e, final_disease_embeddings_e

        if meta_path == 't':
            mirna_embedding = self.dropout(self.m_fc(new_g.ndata['m_sim'][-495:, :].to(self.device)))
            disease_embedding = self.dropout(self.d_fc(new_g.ndata['d_sim'][:383, :].to(self.device)))
            all_mirna_embeddings = []
            all_disease_embeddings = []  # 初始化两个列表
            mirna_R = self.update_attention_A_mi(new_g, train_mirna, train_disease).to_dense()
            disease_R = self.update_attention_A_di(new_g, train_mirna, train_disease).to_dense()
            # aggregate
            for layer in range(self.GCNLayer):
                mirna_embedding = self.activation_layer(torch.mm(mirna_R, disease_embedding))
                disease_embedding = self.activation_layer(torch.mm(disease_R, mirna_embedding))
                all_mirna_embeddings.append(mirna_embedding)
                all_disease_embeddings.append(disease_embedding)
                final_mirna_embeddings = torch.stack(all_mirna_embeddings, dim=1)
                final_mirna_embeddings_t = torch.sum(final_mirna_embeddings, dim=1)
                final_disease_embeddings = torch.stack(all_disease_embeddings, dim=1)
                final_disease_embeddings_t = torch.sum(final_disease_embeddings, dim=1)
            return final_mirna_embeddings_t, final_disease_embeddings_t

        if meta_path == 'g':
            mirna_embedding = self.dropout(self.m_fc(new_g.ndata['m_sim'][-495:, :].to(self.device)))
            disease_embedding = self.dropout(self.d_fc(new_g.ndata['d_sim'][:383, :].to(self.device)))
            all_mirna_embeddings = []
            all_disease_embeddings = []  # 初始化两个列表
            mirna_R = self.update_attention_A_mi(new_g, train_mirna, train_disease).to_dense()
            disease_R = self.update_attention_A_di(new_g, train_mirna, train_disease).to_dense()
            # aggregate
            for layer in range(self.GCNLayer):
                mirna_embedding = self.activation_layer(torch.mm(mirna_R, disease_embedding))
                disease_embedding = self.activation_layer(torch.mm(disease_R, mirna_embedding))
                all_mirna_embeddings.append(mirna_embedding)
                all_disease_embeddings.append(disease_embedding)
                final_mirna_embeddings = torch.stack(all_mirna_embeddings, dim=1)
                final_mirna_embeddings_g = torch.sum(final_mirna_embeddings, dim=1)
                final_disease_embeddings = torch.stack(all_disease_embeddings, dim=1)
                final_disease_embeddings_g = torch.sum(final_disease_embeddings, dim=1)
            return final_mirna_embeddings_g, final_disease_embeddings_g
