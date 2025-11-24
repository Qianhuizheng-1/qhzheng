from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


# '''
# a deprecated aggregation function
# '''


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim_k % num_heads == 0, "dim_k must be divisible by num_heads"
        
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim = dim_k // num_heads
        
        self.linear_q = nn.Linear(dim_q, dim_k)
        self.linear_k = nn.Linear(dim_q, dim_k)
        self.linear_v = nn.Linear(dim_q, dim_v)
        self.linear_out = nn.Linear(dim_v, dim_v)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_v)
        
        self._norm_fact = 1 / sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape for multi-head
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        
        # Reshape and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_v)
        output = self.linear_out(context)
        
        # Add & Norm
        output = self.layer_norm(output + x)
        
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_v)

    def forward(self, x, mask=None):
        # x: batch, n, dim_q
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        
        if mask is not None:
            dist = dist.masked_fill(mask == 0, -1e9)
            
        dist = F.softmax(dist, dim=-1)
        dist = self.dropout(dist)
        
        att = torch.bmm(dist, v)
        att = self.layer_norm(att + x)
        
        return att

def att_inter_agg(att_layer, self_feats, agg_feats, to_feats, embed_dim, a, b, bn, dropout, n, cuda):
    """
    优化的注意力聚合函数
    :param att_layer: 注意力层
    :param self_feats: 自身特征
    :param agg_feats: 聚合特征
    :param to_feats: 目标特征
    :param embed_dim: 嵌入维度
    :param a: 第一个线性变换矩阵
    :param b: 第二个线性变换矩阵
    :param bn: 批归一化层
    :param dropout: dropout层
    :param n: 节点数量
    :param cuda: 是否使用GPU
    :return: 聚合后的特征
    """
    # 合并特征
    neigh_h = torch.cat((agg_feats.transpose(0, 1), to_feats.transpose(0, 1)), dim=0)
    combined = torch.cat((self_feats.repeat(2, 1), neigh_h), dim=1)
    
    # 应用批归一化
    combined = bn(combined)
    
    # 计算注意力分数
    attention = att_layer(combined.mm(a))
    attention = dropout(attention, 0.2, training=True)
    attention = att_layer(attention.mm(b))
    
    # 重塑注意力分数
    attention = torch.cat((attention[0:n, :], attention[n:2 * n, :]), dim=1)
    attention = F.softmax(attention, dim=1)
    
    # 初始化聚合特征
    if cuda:
        aggregated = torch.zeros(size=(n, embed_dim)).cuda()
    else:
        aggregated = torch.zeros(size=(n, embed_dim))
    
    # 使用注意力权重聚合特征
    for r in range(2):
        aggregated += torch.mul(
            attention[:, r].unsqueeze(1).repeat(1, embed_dim),
            neigh_h[r * n:(r + 1) * n, :]
        )
    
    return aggregated.transpose(0, 1)


class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads, dropout=0.1, temperature=1.0):
        super(EnhancedMultiHeadAttention, self).__init__()
        assert dim_k % num_heads == 0, "dim_k must be divisible by num_heads"
        
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim = dim_k // num_heads
        self.temperature = temperature
        
        self.linear_q = nn.Linear(dim_q, dim_k)
        self.linear_k = nn.Linear(dim_q, dim_k)
        self.linear_v = nn.Linear(dim_q, dim_v)
        self.linear_out = nn.Linear(dim_v, dim_v)
        
        # 添加自适应权重
        self.attention_weights = nn.Parameter(torch.ones(num_heads))
        
        # 添加门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_v)
        
        self._norm_fact = 1 / sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 线性投影和重塑
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力，添加温度参数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._norm_fact / self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 使用GELU激活函数
        attn = F.gelu(scores)
        
        # 应用自适应权重
        head_weights = F.softmax(self.attention_weights, dim=0)
        attn = attn * head_weights.view(1, self.num_heads, 1, 1)
        
        attn = self.dropout(attn)
        
        # 应用注意力到值
        context = torch.matmul(attn, v)
        
        # 重塑并应用最终线性层
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_v)
        output = self.linear_out(context)
        
        # 应用门控机制
        gate = self.gate(output)
        output = gate * output + (1 - gate) * x
        
        # 添加残差连接和层归一化
        output = self.layer_norm(output + x)
        
        return output



