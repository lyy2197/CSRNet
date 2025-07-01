import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torch.nn import AvgPool2d

def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim ** .5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i * self.stride
            input = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)

        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    # [batch_size, n_channel, d_model]
    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)

        out, _ = attention(q, k, v)

        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output


class TsNetx(nn.Module):

    def __init__(self, num_classes=3, num_samples=128, num_channels=64, embed_dim_1=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=5, depth=1, attn_drop=0.5, fc_drop=0.5,embed_dim=64, temp_embedding_dim=4):
        super().__init__()
        self.DT_conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(1, embed_dim_1// 4, kernel_size=(1, 8), bias=False),
            nn.BatchNorm2d(embed_dim_1 // 4)
        )

        self.DT_conv2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, embed_dim_1 // 4, kernel_size=(1, 16), bias=False),
            nn.BatchNorm2d(embed_dim_1 // 4)
        )

        self.DT_conv3 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, embed_dim_1 // 4, kernel_size=(1, 32), bias=False),
            nn.BatchNorm2d(embed_dim_1 // 4)
        )

        self.DT_conv4 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, embed_dim_1// 4, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(embed_dim_1 // 4)
        )

        # self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1),groups=embed_dim, bias=False)
        self.SP_conv = nn.Sequential(
            # DepthwiseConv2D
            nn.Conv2d(embed_dim_1, 64, kernel_size=(64, 1), groups=embed_dim_1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(fc_drop),
        )

        self.HOT_conv1 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((0, 1, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 2), groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(64),
        )

        self.HOT_conv2 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 4), groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(64),
        )

        self.HOT_conv3 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 8), groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(64),
        )

        self.HOT_conv4 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(64, 64,kernel_size=(1, 16), groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(64),
        )
        self.block4 = nn.Sequential(
            # nn.ELU(inplace=True),
            nn.Conv2d(256,64,kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(fc_drop)
        )

        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AdaptiveAvgPool1d(temp_embedding_dim)

        self.dropout = nn.Dropout()

        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        )

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (1, 1)),
            nn.BatchNorm2d(temp_embedding_dim),
            nn.ELU()
        )
        self.classify = nn.Linear(embed_dim * temp_embedding_dim, num_classes)

    def forward(self, x):
        x1 = self.DT_conv1(x)  # (16,8,22,1000)
        x2 = self.DT_conv2(x)
        x3 = self.DT_conv3(x)
        x4 = self.DT_conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (16,32,22,1000)
        x = self.SP_conv(x)
        x5 = self.HOT_conv1(x)
        x6 = self.HOT_conv2(x)
        x7 = self.HOT_conv3(x)
        x8 = self.HOT_conv4(x)
        x = torch.cat((x5, x6, x7, x8), dim=1)
        x = self.block4(x)
        x1 = x.squeeze(dim=2)
        x1 = rearrange(x1, 'b d n -> b n d')
        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
        x1 = x1.unsqueeze(dim=2)
        x = self.conv_encoder(x1)
        x = x.reshape(x.size(0), -1)
        out = self.classify(x)
        out = F.sigmoid(self.classify(x))
        return out
