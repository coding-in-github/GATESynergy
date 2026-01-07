import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool as gmp, TransformerConv
from torch_geometric.utils import softmax, to_dense_batch
from torch_geometric.nn.inits import glorot, zeros
import pandas as pd
import numpy as np
import einops

class GlobalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, dropout=0.0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = out_channels // num_heads
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.lin_q = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_k = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_v = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_out = nn.Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_q.weight)
        glorot(self.lin_k.weight)
        glorot(self.lin_v.weight)
        glorot(self.lin_out.weight)
        zeros(self.lin_q.bias)
        zeros(self.lin_k.bias)
        zeros(self.lin_v.bias)
        zeros(self.lin_out.bias)

    def forward(self, x, batch):
        x_dense, mask = to_dense_batch(x, batch)
        B, N, D = x_dense.shape

        q = self.lin_q(x_dense).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        k = self.lin_k(x_dense).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        v = self.lin_v(x_dense).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, N, N]

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            mask_expanded = mask_expanded.expand(B, self.num_heads, N, N)  # [B, heads, N, N]
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = torch.matmul(attn_weights, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_channels)  # [B, N, out_channels]
        out = self.lin_out(out)
        out_sparse = out[mask]  # [total_nodes, out_channels]

        if self.in_channels == self.out_channels:
            out_sparse = out_sparse + x

        return out_sparse


class SGFormer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, dropout=0.0,
                 global_model_type='Transformer', attn_type='multihead'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.global_model_type = global_model_type
        self.attn_type = attn_type
        self.local_branch = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=1,
            dropout=dropout
        )
        if global_model_type == 'Transformer':
            self.global_branch = GlobalAttention(
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported global_model_type: {global_model_type}")

        self.fusion_type = 'weighted_sum'
        if self.fusion_type == 'weighted_sum':
            self.local_weight = nn.Parameter(torch.tensor(0.5))
            self.global_weight = nn.Parameter(torch.tensor(0.5))
        elif self.fusion_type == 'concat':
            self.fusion_proj = nn.Linear(out_channels * 2, out_channels)
        elif self.fusion_type == 'gating':
            self.gate = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.Sigmoid()
            )
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_channels, out_channels)
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.fusion_type == 'concat' and hasattr(self, 'fusion_proj'):
            glorot(self.fusion_proj.weight)
            zeros(self.fusion_proj.bias)
        elif self.fusion_type == 'gating':
            for layer in self.gate:
                if isinstance(layer, nn.Linear):
                    glorot(layer.weight)
                    zeros(layer.bias)

        if self.residual_proj is not None:
            glorot(self.residual_proj.weight)
            zeros(self.residual_proj.bias)

    def forward(self, x, edge_index, batch=None):
        residual1 = x
        if self.residual_proj is not None:
            residual1 = self.residual_proj(residual1)

        local_out = self.local_branch(x, edge_index)
        global_out = self.global_branch(x, batch)

        if self.fusion_type == 'weighted_sum':
            fused_out = self.local_weight * local_out + self.global_weight * global_out
        elif self.fusion_type == 'concat':
            concatenated = torch.cat([local_out, global_out], dim=-1)
            fused_out = self.fusion_proj(concatenated)
        elif self.fusion_type == 'gating':
            concatenated = torch.cat([local_out, global_out], dim=-1)
            gate_weights = self.gate(concatenated)
            fused_out = gate_weights * local_out + (1 - gate_weights) * global_out
        else:
            fused_out = local_out + global_out

        out = fused_out + residual1
        out = self.norm1(out)
        out = self.dropout_layer(out)
        return out

class ResidualGatedNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        gate = self.gate(x)
        transformed = self.transform(x)
        return x + gate * transformed

class EfficientAdditiveAttention(nn.Module):
    def __init__(self, in_dims, token_dim, num_heads=1, use_softmax=True):
        super().__init__()
        assert num_heads >= 1, "num_heads must be >= 1"
        self.in_dims = in_dims
        self.head_dim = token_dim
        self.num_heads = num_heads
        self.total_dim = self.head_dim * self.num_heads
        self.use_softmax = use_softmax
        self.to_query = nn.Linear(in_dims, self.total_dim)
        self.to_key   = nn.Linear(in_dims, self.total_dim)
        self.w_g = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.scale_factor = self.head_dim ** -0.5
        self.Proj  = nn.Linear(self.total_dim, self.total_dim)
        self.final = nn.Linear(self.total_dim, self.head_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)
        nn.init.xavier_uniform_(self.Proj.weight)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.xavier_uniform_(self.w_g)

        if self.to_query.bias is not None:
            nn.init.constant_(self.to_query.bias, 0)
        if self.to_key.bias is not None:
            nn.init.constant_(self.to_key.bias, 0)
        if self.Proj.bias is not None:
            nn.init.constant_(self.Proj.bias, 0)
        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, 0)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.to_query(x).view(B, N, self.num_heads, self.head_dim)   # [B,N,H,Dh]
        k = self.to_key(x).view(B, N, self.num_heads, self.head_dim)     # [B,N,H,Dh]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        query_weight = torch.einsum('bnhd,hd->bnh', q, self.w_g)
        A = query_weight * self.scale_factor                          # [B,N,H]
        if self.use_softmax:
            A = F.softmax(A, dim=1)
        else:
            A = F.normalize(A, dim=1)

        G = torch.einsum('bnh,bnhd->bhd', A, q)
        G_exp = G.unsqueeze(1).expand(B, N, self.num_heads, self.head_dim)     # [B,N,H,Dh]
        out = (G_exp * k).reshape(B, N, self.total_dim)                         # [B,N,H*Dh]
        out = self.Proj(out) + q.reshape(B, N, self.total_dim)                  # [B,N,H*Dh]
        out = self.final(out)                                                   # [B,N,Dh=token_dim]
        return out

class ResidualSGFormer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1,
                 global_model_type='Transformer', attn_type='multihead'):
        super(ResidualSGFormer, self).__init__()
        self.conv = SGFormer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=heads,
            dropout=dropout,
            global_model_type=global_model_type,
            attn_type=attn_type
        )

        self._initialize_weights()

    def _initialize_weights(self):
        pass

    def forward(self, x, edge_index, batch=None):
        out = self.conv(x, edge_index, batch)
        return out

class DNN(nn.Module):
    def __init__(self, layers, dropout=0.2):
        super(DNN, self).__init__()

        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))] + [nn.BatchNorm1d(layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.dropout = nn.Dropout(p=dropout)
        self.activate = nn.LeakyReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        step = int(len(self.dnn_network) / 2)
        for i in range(step):
            linear = self.dnn_network[i]
            batchnorm = self.dnn_network[i + step]
            x = self.dropout(x)
            x = linear(x)
            x = batchnorm(x)
            x = self.activate(x)

        x = self.dropout(x)
        return x

class CellFeatureProcessor(nn.Module):
    def __init__(self, input_dim=954, output_dim=128, intermediate_dims=[512, 256]):
        super().__init__()

        dims = [input_dim] + intermediate_dims + [output_dim]
        self.dimension_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.dimension_layers.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(0.1)
                )
            )

        self.residual_gated_nets = nn.ModuleList()
        for dim in intermediate_dims + [output_dim]:
            self.residual_gated_nets.append(ResidualGatedNet(dim))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.normalize(x, 2, 1)
        for i, (dim_layer, gated_net) in enumerate(zip(self.dimension_layers, self.residual_gated_nets)):
            x = dim_layer(x)
            x = gated_net(x)

        return x


class GATESynergy(torch.nn.Module):
    def __init__(self, drug_dim=64, cell_dim=954, output_dim=128, dropout=0.1, attention_token_dim=256, global_model_type='Transformer', stacked_size=128):

        super(GATESynergy, self).__init__()

        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.drug_conv1 = ResidualSGFormer(78, drug_dim * 2, heads=1, dropout=dropout,
                                           global_model_type=global_model_type)  # (78 -> 128)
        self.drug_conv2 = ResidualSGFormer(drug_dim * 2, drug_dim * 4, heads=1,
                                           dropout=dropout, global_model_type=global_model_type)  # (128 -> 256)
        self.drug_conv3 = ResidualSGFormer(drug_dim * 4, drug_dim * 8, heads=1,
                                           dropout=dropout, global_model_type=global_model_type)  # (256 -> 512)
        self.drug_fc_g1 = torch.nn.Linear(drug_dim * 8, drug_dim * 4)  # (512 -> 256)
        self.drug_fc_g2 = torch.nn.Linear(drug_dim * 4, output_dim)  # (256 -> 128)
        self.cell_processor = CellFeatureProcessor(
            input_dim=cell_dim,  # 954
            output_dim=output_dim,  # 128
            intermediate_dims=[512, 256]
        )
        self.attention = EfficientAdditiveAttention(
            in_dims=stacked_size,
            token_dim=attention_token_dim,
            num_heads=1
        )
        self.feature_aggregation = nn.Sequential(
            nn.Linear(attention_token_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        hidden_units = [256, 64]
        dnn_dropout = 0.2
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        x1 = self.drug_conv1(x1, edge_index1, batch1)
        x1 = self.drug_conv2(x1, edge_index1, batch1)
        x1 = self.drug_conv3(x1, edge_index1, batch1)
        x1 = gmp(x1, batch1)  # (batch_size, 512)
        x1 = self.drug_fc_g1(x1)  # (batch_size, 256)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)
        x1 = self.drug_fc_g2(x1)  # (batch_size, 128)
        x1 = self.dropout(x1)
        x2 = self.drug_conv1(x2, edge_index2, batch2)
        x2 = self.drug_conv2(x2, edge_index2, batch2)
        x2 = self.drug_conv3(x2, edge_index2, batch2)
        x2 = gmp(x2, batch2)  # (batch_size, 512)
        x2 = self.drug_fc_g1(x2)  # (batch_size, 256)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)
        x2 = self.drug_fc_g2(x2)  # (batch_size, 128)
        x2 = self.dropout(x2)
        cell_vector = self.cell_processor(cell)  # (batch_size, 954) -> (batch_size, 128)
        sparse_embeds = torch.stack([x1, x2, cell_vector], dim=1)  # (batch_size, 3, 128)
        attention_output = self.attention(sparse_embeds)  # (batch_size, 3, token_dim)
        attention_flat = attention_output.view(attention_output.size(0), -1)  # (batch_size, 3 * token_dim)
        aggregated_features = self.feature_aggregation(attention_flat)  # (batch_size, 256)
        dnn_x = self.dnn_network(aggregated_features)  # (batch_size, 64)
        final = self.dense_final(dnn_x)  # (batch_size, 1)
        outputs = torch.sigmoid(final.squeeze(1))  # (batch_size,)
        return outputs
