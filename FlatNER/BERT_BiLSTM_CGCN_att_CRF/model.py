# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertModel
from torchcrf import CRF
from sklearn.metrics.pairwise import cosine_similarity
from config import *

class CGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CGCN, self).__init__()
        self.gcn1 = GCNConv(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)  # 仅在第一个GCN层后加入批归一化
        self.gcn2 = GCNConv(out_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(self.bn1(x))  # 批归一化放在ReLU之前
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, ltp):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.bn_lstm = nn.BatchNorm1d(2 * HIDDEN_SIZE)  # 批归一化LSTM输出
        
        self.cgcn_seq = CGCN(EMBEDDING_DIM, HIDDEN_SIZE)
        self.cgcn_dep = CGCN(EMBEDDING_DIM, HIDDEN_SIZE)
        
        self.seq_feats_linear = nn.Linear(HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        self.dep_feats_linear = nn.Linear(HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        
        # 融合线性层
        self.fusion_linear = nn.Linear(3 * 2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, word_embeddings, mask):
        lstm_out, _ = self.lstm(word_embeddings)
        batch_size, seq_len, hidden_size = lstm_out.size()
        lstm_out = lstm_out.contiguous().view(-1, hidden_size)
        lstm_out = self.bn_lstm(lstm_out)
        lstm_out = lstm_out.view(batch_size, seq_len, hidden_size)
        return lstm_out
    

    def _sample_subgraph(self, edge_index, embeddings=None, valid_mask=None, hop=1, sample_type='attention', top_k=1):
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
    
        if hop == 1:
            return edge_index

        elif hop == 2:
            if sample_type == 'attention' and embeddings is not None:
                if valid_mask is not None:
                    embeddings = embeddings[valid_mask]
                num_nodes, _ = embeddings.shape

                # 定义用于计算注意力的投影矩阵
                W_q = nn.Linear(embeddings.size(1), embeddings.size(1)).to(embeddings.device)
                W_k = nn.Linear(embeddings.size(1), embeddings.size(1)).to(embeddings.device)
            
                # 计算查询和键的投影
                queries = W_q(embeddings)
                keys = W_k(embeddings)
            
                # 计算注意力得分
                attention_scores = torch.matmul(queries, keys.T) / (embeddings.size(1) ** 0.5)
                attention_weights = F.softmax(attention_scores, dim=-1)

                # 为每个节点选取 top-1 注意力权重最高的节点
                sampled_edges = []
                for node in range(num_nodes):
                    # 选取当前节点注意力权重最高的节点作为邻居
                    top_idx = torch.argmax(attention_weights[node]).item()
                    if top_idx != node:  # 确保不会连接到自己
                        sampled_edges.append((node, top_idx))

                # 构建与 edge_index 形状相同的输出，初始化为原始边
                padded_sampled_edges = edge_index.clone()
                for i, (src, dst) in enumerate(sampled_edges):
                    if i < padded_sampled_edges.size(0):
                        padded_sampled_edges[i] = torch.tensor([src, dst], dtype=torch.long).to(embeddings.device)

                return padded_sampled_edges

            else:
                return edge_index


    def forward(
        self, input, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2,
        word_vectors, input_texts, dependency_masks=None, word_vectors_masks=None, return_logits=False
    ):
        device = input.device

        # 获取 word_embeddings
        word_embeddings = self.bert(input, mask).last_hidden_state
        lstm_out = self._get_lstm_feature(word_embeddings, mask)

        word_vectors = [vec.to(device) for vec in word_vectors]
        dependency_masks = [dep_mask.to(device) for dep_mask in dependency_masks]
        word_vectors_masks = [mask.to(device) for mask in word_vectors_masks]

        # 获取序列图特征
        seq_feats = [
            self.cgcn_seq(word_embeddings[idx], torch.tensor(seq_edge, dtype=torch.long).t().contiguous().to(device)) 
            for idx, seq_edge in enumerate(sequence_edges)
        ]

        # 使用基于注意力机制的二阶子图采样
        dep_feats_hop1 = [
            self.cgcn_dep(word_vectors[idx].clone().detach(), dep_edge.t().to(device))
            for idx, dep_edge in enumerate(dependency_edges_hop1)
        ]
        dep_feats_hop2 = [
            self.cgcn_dep(word_vectors[i].clone().detach(), dep_edge.t().contiguous().to(device))
            for i, dep_edge in enumerate(dependency_edges_hop2)
        ]



        # 计算合并特征
        dep_feats = [
            torch.cat([hop1, hop2], dim=-1) * dep_mask.float().unsqueeze(-1)
            for hop1, hop2, dep_mask in zip(dep_feats_hop1, dep_feats_hop2, dependency_masks)
        ]
        seq_feats = [self.seq_feats_linear(feat) for feat in seq_feats]

        word_vectors = [
            vec * word_mask.unsqueeze(-1) for vec, word_mask in zip(word_vectors, word_vectors_masks)
        ]

        combined_feats = [
            self.fusion_linear(torch.cat([lstm_out[idx], seq_feats[idx], dep_feats[idx]], dim=-1))
            for idx in range(len(seq_feats))
        ]
        
        logits = self.linear(torch.stack(combined_feats))
        
        if return_logits:
            return logits
        decoded_tags = self.crf.decode(logits, mask)
        return decoded_tags

    def loss_fn(self, input, target, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2, word_vectors, input_texts, dependency_masks=None, word_vectors_masks=None):
        logits = self.forward(input, mask, sequence_edges, dependency_edges_hop1, dependency_edges_hop2, word_vectors, input_texts, dependency_masks, word_vectors_masks, return_logits=True)
        loss = -self.crf(logits, target, mask=mask, reduction='mean')
        return loss
