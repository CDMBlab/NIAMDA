import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon import nn as ng

if torch.cuda.is_available():
    context = torch.device('cuda')
else:
    context = torch.device('cpu')

# from layers import MultiHeadGATLayer, Metasubgraph_semantic_gat # 从layers层导入
from gcn import NIE_GCN, NIE_GCN_Metasubgraph


# from xiaorong import GCN


# 语义层注意力
# 主要用于修改数据，作比较模型，这里主要有两个地方：1.hidden_size; 2.beta
# 1是语义层向量q的维度变化对参数的影响；2是beta用于语义层贡献相同时的结果
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=512):  # in_size=out_dim * num_heads =512 8*64
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):  # [x, 2, 512] ，X是节点个数
        w = self.project(z).mean(0)  # 权重矩阵，获得元路径的重要性 [2, 1]
        # beta = torch.softmax(w, dim=0)
        beta = torch.sigmoid(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # [x,2,1]
        # delta = beta * z
        return (beta * z).sum(1)  # [x, 512]


class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        # 初始化可学习的权重矩阵 W
        self.W = nn.Parameter(torch.randn(feature_size, feature_size))

        # 激活函数
        self.activation = nn.Sigmoid()

    def forward(self, h_diseases, h_mirnas):
        # 双线性解码器的计算过程
        # h_diseases 和 W 做矩阵乘法，然后与 h_mirnas 对应元素相乘
        results = torch.matmul(h_diseases, self.W) * h_mirnas

        # 对结果在特征维度求和
        results_mask = results.sum(dim=1).unsqueeze(1)

        # 使用 sigmoid 激活函数将结果映射到 [0, 1] 区间
        return self.activation(results_mask)


##wfy
class HGANMDA_multi(nn.Module):  # 传入的G是g0_train：三元异质图的训练子图
    def __init__(self, G, meta_paths_list, train_mirna, train_disease, train_mirna_c, train_disease_c, train_mirna_e,
                 train_disease_e, train_mirna_t, train_disease_t, train_mirna_g,
                 train_disease_g, num_heads, num_diseases, num_mirnas, d_sim_dim, m_sim_dim, out_dim,
                 dim, GCNLayer, beta, device, dropout):
        super(HGANMDA_multi, self).__init__()

        self.G = G

        self.train_mirna = train_mirna
        self.train_disease = train_disease
        self.train_mirna_c = train_mirna_c
        self.train_disease_c = train_disease_c
        self.train_disease_e = train_disease_e
        self.train_mirna_e = train_mirna_e
        self.train_mirna_t = train_mirna_t
        self.train_disease_t = train_disease_t
        self.train_mirna_g = train_mirna_g
        self.train_disease_g = train_disease_g
        self.meta_paths = meta_paths_list
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.dim = dim
        self.GCNLayer = GCNLayer
        self.beta = beta
        self.device = device
        self.num_mirnas = num_mirnas

        self.gcn = NIE_GCN(G, train_disease, train_mirna, dim, GCNLayer, beta, dropout, device)
        self.meta_gcn = NIE_GCN_Metasubgraph(G, train_disease, train_mirna, dim, GCNLayer, beta, dropout, device)
        # self.gcn = GCN(G, num_mirnas, num_diseases, dim, GCNLayer, dropout, device)
        # self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope) #输入：图、特征注意力向量、头数、丢弃率 ，在layers中leckrelu
        self.heads = nn.ModuleList()

        self.metapath_layers = nn.ModuleList()  # 四个元路径层，每个元路径需要attn_heads个元路径注意力层
        # for i in range(self.num_heads):
        # self.metapath_layers.append(Metasubgraph_semantic_gat(G, feature_attn_size, out_dim, dropout, slope))

        self.dropout = nn.Dropout(dropout)
        self.m_fc = nn.Linear(dim + m_sim_dim, out_dim)  # 多头miRNA特征降维映射矩阵
        self.d_fc = nn.Linear(dim + d_sim_dim, out_dim)  # 多头disease降维映射矩阵
        # self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads) #语意注意力层,修改输出
        self.semantic_attention = SemanticAttention(in_size=out_dim)  # 语意注意力层

        self.h_fc = nn.Linear(out_dim, out_dim)  # 全链接
        # self.predict = nn.Linear(out_dim * 2, 1)
        self.Bilinear = BilinearDecoder(out_dim)
        ###wfy：多分类的预测函数
        # self.predict = nn.Linear(out_dim * 2, 5) #有五类
        ###

    def forward(self, G, G0, train_mirna, train_disease, train_mirna_c, train_disease_c, train_mirna_e, train_disease_e,
                train_mirna_t, train_disease_t, train_mirna_g, train_disease_g, diseases, mirnas):  # 模型中的数据流动过程 G是g0_train,G0是G0,疾病和基因边的节点列表
        index1 = 0
        # multi_md = 1
        for meta_path in self.meta_paths:  # 对于四种元路径分别 ['md', 'dm', 'ml', 'dl'][c,e,t,g]
            if meta_path == 'md' or meta_path == 'dm':
                # 元路径为md和dm时，获得的聚合特征0-382是疾病特征。383-877是miRNA特征
                if index1 == 0:  # 只进行一次GAT，更新m和d的特征
                    mirna_embedding, disease_embedding = self.gcn(G, train_mirna, train_disease)
                    index1 = 1
            elif meta_path == 'c':
                c_edges = G0.filter_edges(lambda edges: edges.data['multi_label'] == 1)  # 选元子图的边
                g_c = G0.edge_subgraph(c_edges, preserve_nodes=True)
                mirna_embedding_c, disease_embedding_c = self.meta_gcn(g_c, meta_path, train_mirna_c, train_disease_c)
            elif meta_path == 'e':
                e_edges = G0.filter_edges(lambda edges: edges.data['multi_label'] == 2)
                g_e = G0.edge_subgraph(e_edges, preserve_nodes=True)
                mirna_embedding_e, disease_embedding_e = self.meta_gcn(g_e, meta_path, train_mirna_e, train_disease_e)
            elif meta_path == 't':
                t_edges = G0.filter_edges(lambda edges: edges.data['multi_label'] == 3)
                g_t = G0.edge_subgraph(t_edges, preserve_nodes=True)
                mirna_embedding_t, disease_embedding_t = self.meta_gcn(g_t, meta_path, train_mirna_t, train_disease_t)
            elif meta_path == 'g':
                g_edges = G0.filter_edges(lambda edges: edges.data['multi_label'] == 4)
                g_g = G0.edge_subgraph(g_edges, preserve_nodes=True)
                mirna_embedding_g, disease_embedding_g = self.meta_gcn(g_g, meta_path, train_mirna_g, train_disease_g)

        semantic_embeddings1 = torch.stack(
            (disease_embedding, disease_embedding_c, disease_embedding_e, disease_embedding_t, disease_embedding_g),
            dim=1)  # 语意嵌入  tensor(383,2,512)
        h1 = self.semantic_attention(semantic_embeddings1)  # (495*512)
        semantic_embeddings2 = torch.stack(
            (mirna_embedding, mirna_embedding_c, mirna_embedding_e, mirna_embedding_t, mirna_embedding_g), dim=1)
        h2 = self.semantic_attention(semantic_embeddings2)  # （495*512)

        # 将经过语义层注意力得到的疾病特征和miRNA特征，和原来的疾病特征和miRNA特征连接
        h_d = torch.cat((h1, self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        h_m = torch.cat((h2, self.G.ndata['m_sim'][self.num_diseases:878]), dim=1)
        h_m = self.dropout(F.elu(self.m_fc(h_m)))  # （495，1007）->（495,64）
        h_d = self.dropout(F.elu(self.d_fc(h_d)))  # (383,895)->（383,64）

        h = torch.cat((h_d, h_m), dim=0)  # （878,64）
        h = self.dropout(F.elu(self.h_fc(h)))  # 做一个全链接 （878,64)-(878,64)

        # 获取训练边或测试边的点的特征
        h_diseases = h[diseases]  # 样本边disease特征;(17376,64)
        h_mirnas = h[mirnas]  # 样本边mirna特征(17376,64)
        # 全连接层得到结果
        # h_concat = torch.cat((h_diseases, h_mirnas), 1)  # (17376,128)
        # predict_score = torch.sigmoid(self.predict(h_concat))  # (17376,128)->(17376,128*2)->(17376,1)
        ##wfy多分类激活函数，应该为softmax
        # predict_score = torch.softmax(self.predict(h_concat), dim=1)   # (17376,128)->(17376,128*2)->(17376,num_classes)
        ##
        predict_score = self.Bilinear(h_diseases, h_mirnas)
        return predict_score
