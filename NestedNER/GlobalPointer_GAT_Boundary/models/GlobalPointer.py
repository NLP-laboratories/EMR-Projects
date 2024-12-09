import sys

sys.path.append("../")
from common.utils import Preprocessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch_geometric.nn import LayerNorm
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataMaker(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """生成喂入模型的数据，支持实体类别"""
        ent_type_size = len(ent2id)
        all_inputs = []

        for sample in datas:
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length'
            )

            labels = None
            span_features = []  # 新增 span_features 列表
            if data_type != "predict":
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"]
                )
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    labels[ent2id[label], start, end] = 1
                    span_features.append((start, end, ent2id[label]))  # 添加起止位置和实体类别

            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if labels is not None:
                labels = torch.tensor(inputs["labels"]).long()

            # 将包含实体类别的 span_features 加入 sample_input
            sample_input = (sample, input_ids, attention_mask, token_type_ids, labels, span_features)
            all_inputs.append(sample_input)

        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []
        span_features_list = []  # 新增 span_features 列表

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "predict":
                labels_list.append(sample[4])
            span_features_list.append(sample[5])  # 添加包含实体类别的 span_features

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0) if data_type != "predict" else None

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels, span_features_list


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, hidden_dim=128, use_gat=True, RoPE=True, use_gaussian_kernel=False):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE
        self.use_gat = use_gat
        self.use_gaussian_kernel = use_gaussian_kernel

        

        # GAT Module
        if self.use_gat:
            self.gat1 = GATConv(self.hidden_size, hidden_dim, heads=2, concat=True)
            self.norm = LayerNorm(hidden_dim * 2)
            self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)

        # Dense Layer
        gat_output_dim = hidden_dim * 2 if use_gat else 0
        self.dense = nn.Linear(self.hidden_size + self.hidden_size + gat_output_dim, self.ent_type_size * self.inner_dim * 2)

        # Boundary Enhancement Modules using MLP
        self.start_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Compression Layer to avoid feature explosion
        self.compression_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(seq_len, output_dim)
        embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return embeddings.to(self.device)

    def construct_batch_graphs(self, span_features_batch, hidden_states_batch):
        """
        构造批量图，仅使用 span 内的 token 均值特征作为节点特征。
        
        Args:
            span_features_batch (list): 每个样本的 span 特征，格式为 [(start, end, ent_type), ...]。
            hidden_states_batch (torch.Tensor): 每个样本的隐层特征 [batch_size, seq_len, hidden_size]。
        
        Returns:
            Batch: 批量图数据。
        """
        all_graphs = []

        for span_features, hidden_states in zip(span_features_batch, hidden_states_batch):
            nodes = []  # 节点特征
            edges = []  # 边索引

            # 构造节点特征，仅计算 span 内 token 的均值特征
            for (start, end, _) in span_features:
                if start < hidden_states.size(0) and end < hidden_states.size(0):  # 检查索引合法性
                    # span 内 token 均值特征
                    span_feature = hidden_states[start:end + 1].mean(dim=0)
                    nodes.append(span_feature)

            # 构造边关系 (嵌套关系)
            num_spans = len(span_features)
            for i in range(num_spans):
                for j in range(num_spans):
                    # 判断嵌套关系
                    start_i, end_i, _ = span_features[i]
                    start_j, end_j, _ = span_features[j]
                    if start_i <= start_j and end_j <= end_i and (start_i, end_i) != (start_j, end_j):
                        # i 包含 j，添加 i -> j 的边
                        edges.append((i, j))
                    elif start_j <= start_i and end_i <= end_j and (start_i, end_i) != (start_j, end_j):
                        # j 包含 i，添加 j -> i 的边
                        edges.append((j, i))

            # 如果没有节点，构造空图
            if len(nodes) == 0:
                graph = Data(
                    x=torch.zeros((1, hidden_states.size(-1)), device=self.device),  # 空节点
                    edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device)  # 空边
                )
            else:
                nodes = torch.stack(nodes, dim=0).to(self.device)  # 节点特征 [num_nodes, hidden_size]
                edges = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous() if edges else \
                    torch.empty((2, 0), dtype=torch.long, device=self.device)  # 边索引
                graph = Data(x=nodes, edge_index=edges)

            all_graphs.append(graph)

        # 转换为批量图
        return Batch.from_data_list(all_graphs)
    


    def boundary_enhance(self, last_hidden_state, span_features_batch):
        """
        使用 MLP 增强边界 token 特征（start 和 end），并替换对应特征。
        """
        seq_len, hidden_size = last_hidden_state.size(1), last_hidden_state.size(2)
        enhanced_hidden_state = last_hidden_state.clone()

        for b, spans in enumerate(span_features_batch):
            for start, end, _ in spans:
                if start < seq_len and end < seq_len:
                    # 提取起始和结束位置的特征
                    start_feature = last_hidden_state[b, start]  # [hidden_size]
                    end_feature = last_hidden_state[b, end]      # [hidden_size]

                    # 使用 MLP 增强特征
                    start_enhanced = self.start_mlp(start_feature)  # [hidden_size]
                    end_enhanced = self.end_mlp(end_feature)        # [hidden_size]

                    # 替换增强后的起始和结束位置特征
                    enhanced_hidden_state[b, start] = start_enhanced
                    enhanced_hidden_state[b, end] = end_enhanced

        return enhanced_hidden_state



    def forward(self, input_ids, attention_mask, token_type_ids, span_features_batch=None):
        self.device = input_ids.device

        # Step 1: Transformer Encoder Output
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]  # Shape: [batch_size, seq_len, hidden_size]

        # Step 2: Boundary Enhancement Features
        if span_features_batch is not None:
            boundary_features = self.boundary_enhance(last_hidden_state, span_features_batch)
        else:
            boundary_features = torch.zeros_like(last_hidden_state)

        # Step 3: GAT Feature Extraction
        if self.use_gat and span_features_batch is not None:
            # Construct Graph
            batch_graph = self.construct_batch_graphs(span_features_batch, last_hidden_state)
            batch_graph = batch_graph.to(self.device)

            # GAT Layers
            gat_output = self.gat1(batch_graph.x, batch_graph.edge_index)
            gat_output = F.relu(gat_output)
            gat_output = self.norm(gat_output)
            gat_output = self.gat2(gat_output, batch_graph.edge_index)
            gat_output = F.relu(gat_output)

            # Split GAT Output by Batch
            enhanced_features = torch.split(gat_output, batch_graph.batch.bincount().tolist())
            enhanced_features = torch.cat([feat.mean(dim=0, keepdim=True) for feat in enhanced_features], dim=0)

            # Token-Level GAT Features
            seq_len = last_hidden_state.size(1)
            token_enhanced_features = torch.zeros_like(last_hidden_state[..., :enhanced_features.size(-1)])
            for b, spans in enumerate(span_features_batch):
                for start, end, _ in spans:
                    if start < seq_len and end < seq_len:
                        token_enhanced_features[b, start:end + 1] += enhanced_features[b]
        else:
            token_enhanced_features = torch.zeros_like(last_hidden_state)

        # Step 4: Combine All Features
        combined_features = torch.cat([
            last_hidden_state,
            boundary_features,
            token_enhanced_features
        ], dim=-1)

        # Step 5: Dense Layer for qw and kw
        outputs = self.dense(combined_features)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # Step 6: RoPE Encoding (Optional)
        if self.RoPE:
            batch_size, seq_len = last_hidden_state.shape[:2]
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1).reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1).reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # Step 7: Calculate Logits
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # Padding Mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(logits.shape)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # Lower Triangle Mask
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5