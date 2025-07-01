import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import BatchNorm3d
from timm.models.layers import DropPath, to_2tuple


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, patch_size):
        H, W = patch_size, patch_size
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SMFFN(nn.Module):
    """ Self Modulated Feed Forward Network (SM-FFN) in "Hybrid Spectral Denoising Transformer with Learnable Query"
    """

    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w_3 = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, input):
        x = self.w_1(input)
        x = F.gelu(x)
        x1 = self.w_2(x)

        x = self.w_3(input)
        x, w = torch.chunk(x, 2, dim=-1)
        x2 = x * torch.sigmoid(w)

        return x1 + x2


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads  # 头的个数
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector,thats why *3 对QVK三组向量先进行线性操作
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)
        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的头数

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # QK向量先做点乘 来计算相关性 然后除以缩放因子

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        # softmax结果和value向量相乘 得到最终结果
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax

        # 重新整理维度
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block

        out = self.nn1(out)

        out = self.do1(out)
        return out


def repeat(x):
    if isinstance(x, (tuple, list)):
        return x
    return [x] * 3


class S3Conv(nn.Module):
    # deep wise then point wise
    def __init__(self, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            # nn.Conv3d(out_ch, out_ch, (1, k[1], k[2]), 1, (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            # nn.LeakyReLU(),
        )
        self.pw_conv = nn.Conv3d(in_ch, out_ch, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        return x1 + x2


def PlainConv(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', S3Conv(in_ch, out_ch, 3, 1, 1, bias=False)),
        ('bn', BatchNorm3d(out_ch)),
    ]))


class sert_ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        super(sert_ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.low = 16
        self.subnet = nn.Sequential(

            nn.Linear(num_feat, self.low),
            # nn.Linear(num_feat, num_feat // squeeze_factor),
            # nn.ReLU(inplace=True)
        )
        self.upnet = nn.Sequential(
            nn.Linear(self.low, num_feat),
            # nn.Linear(num_feat, num_feat),
            nn.Sigmoid())
        self.mb = torch.nn.Parameter(torch.randn(self.low, memory_blocks))
        self.low_dim = self.low

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        b, n, c = x.shape
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2 # x-[64, 121, 128],y2-[64, 1, 128]
        # print(out.shape)
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        return out


def get_seqlen_and_mask(input_resolution, window_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask

class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x

class AggregatedAttention_rev(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size1=3, window_size2=5, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        assert window_size1 % 2 == 1, "window size must be odd"
        self.window_size1 = window_size1
        self.local_len1 = window_size1 ** 2
        assert window_size2 % 2 == 1, "window size must be odd"
        self.window_size2 = window_size2
        self.local_len2 = window_size2 ** 2

        if fixed_pool_size is None:
            self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        else:
            assert fixed_pool_size < min(input_resolution), \
                f"The fixed_pool_size {fixed_pool_size} should be less than the shorter side of input resolution {input_resolution} to ensure pooling works correctly."
            self.pool_H, self.pool_W = fixed_pool_size, fixed_pool_size
        self.pool_len = self.pool_H * self.pool_W

        self.unfold1 = nn.Unfold(kernel_size=window_size1, padding=window_size1 // 2, stride=1)
        self.unfold2 = nn.Unfold(kernel_size=window_size2, padding=window_size2 // 2, stride=1)
        # self.temperature = nn.Parameter(
        #     torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.query_embedding = nn.Parameter(
        #     nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local1 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len1), mean=0, std=0.0004))
        self.relative_pos_bias_local2 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len2), mean=0, std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length1, padding_mask1 = get_seqlen_and_mask(input_resolution, window_size1)
        local_seq_length2, padding_mask2 = get_seqlen_and_mask(input_resolution, window_size2)
        # self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length1.numpy() + local_seq_length2.numpy() + self.pool_len)),
        #                      persistent=False)
        self.register_buffer("padding_mask1", padding_mask1, persistent=False)
        self.register_buffer("padding_mask2", padding_mask2, persistent=False)


        # dynamic_local_bias:
        self.learnable_tokens1 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len1), mean=0, std=0.02))
        self.learnable_bias1 = nn.Parameter(torch.zeros(num_heads, 1, self.local_len1))
        self.learnable_tokens2 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len2), mean=0, std=0.02))
        self.learnable_bias2 = nn.Parameter(torch.zeros(num_heads, 1, self.local_len2))
        self.outlook = OutlookAttention(dim=self.dim, num_heads=self.num_heads)

    def forward(self, x, patch_size):
        H, W = patch_size, patch_size
        B, N, C = x.shape
        # print("x:", x.shape)

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = q_norm

        # Generate unfolded keys and values and l2-normalize them
        k_local1, v_local1 = self.kv(x).chunk(2, dim=-1)
        k_local1 = F.normalize(k_local1.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local1 = torch.cat([k_local1, v_local1], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local1, v_local1 = self.unfold1(kv_local1).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len1, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local1 = ((q_norm_scaled.unsqueeze(-2) @ k_local1).squeeze(-2)).masked_fill(self.padding_mask1, float('-inf'))

        # Generate unfolded keys and values and l2-normalize them
        k_local2, v_local2 = self.kv(x).chunk(2, dim=-1)
        k_local2 = F.normalize(k_local2.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local2 = torch.cat([k_local2, v_local2], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local2, v_local2 = self.unfold2(kv_local2).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len2, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local2 = ((q_norm_scaled.unsqueeze(-2) @ k_local2).squeeze(-2) ).masked_fill(self.padding_mask2, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1)

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        # attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = torch.cat([attn_local1, attn_local2, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local1, attn_local2, attn_pool = torch.split(attn, [self.local_len1, self.local_len2, self.pool_len], dim=-1)

        # x_local1 = (((q_norm @ self.learnable_tokens1) + self.learnable_bias1 + attn_local1).unsqueeze(
        #     -2) @ v_local1.transpose(-2, -1)).squeeze(-2)
        x_local2 = (((q_norm @ self.learnable_tokens2) + self.learnable_bias2 + attn_local2).unsqueeze(
            -2) @ v_local2.transpose(-2, -1)).squeeze(-2)

        # 消 outlooker
        x_hw = x.view(B, H, W, C)
        outlooker1 = self.outlook(x_hw)  # [B, H, W, C]
        # Step C: [B, H, W, C] → [B, N, C] → [B, N, num_heads, head_dim] → [B, num_heads, N, head_dim]
        outlooker1 = outlooker1.view(B, N, C).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Step D: 将其转化为 attention logits （仿照 q_norm @ tokens1）
        outlooker1 = torch.matmul(outlooker1, self.learnable_tokens1)  # [B, num_heads, N, local_len1]
        # outlooker2 = torch.matmul(outlooker1, self.learnable_tokens2)
        # Step E: 聚合局部特征
        # print("out:", outlooker1.shape)
        # print("attn:", attn_local1.shape)
        x_local1 = ((outlooker1 + attn_local1).unsqueeze(-2) @ v_local1.transpose(-2, -1)).squeeze(-2)
        # x_local2 = ((outlooker2 + attn_local2).unsqueeze(-2) @ v_local2.transpose(-2, -1)).squeeze(-2)

        # 消 learnable_tokens
        # x_local1 = (attn_local1.unsqueeze(
        #     -2) @ v_local1.transpose(-2, -1)).squeeze(-2)
        # x_local2 = (attn_local2.unsqueeze(
        #     -2) @ v_local2.transpose(-2, -1)).squeeze(-2)


        x_pool = attn_pool @ v_pool
        x = (x_local1 + x_local2 + x_pool).transpose(1, 2).reshape(B, N, C)

        # Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # Excitation
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Fully Connected
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class FASA(nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int):
        super().__init__()
        self.q = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0),
                               nn.BatchNorm2d(dim))

        self.local_mixer = sert_ChannelAttention(dim)
        self.patchsize = patch_size
        self.aggatt1 = AggregatedAttention_rev(dim=dim, input_resolution=to_2tuple(patch_size),
                                window_size1=3, window_size2=5, num_heads=num_heads, fixed_pool_size=2)
        self.lamuda = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
        self.fc = nn.Linear(dim*2, dim)
        self.depthwise = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1
                                   ,groups=dim))
        self.pointwise = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1))
        self.mixer = nn.Sequential(nn.Conv2d(dim*2, dim, 1, 1, 0),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(),
                                   nn.Conv2d(dim, dim, 3, 1, 1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU(),
                                   )
        self.se = SEBlock(dim, reduction=16)


    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        B, C, H, W = x.size()
        q_local = self.q(x)
        local_feat = self.local_mixer(q_local)
        global_feat = x.flatten(2).permute(0, 2, 1)
        global_feat = self.aggatt1(global_feat, self.patchsize)
        global_feat = global_feat.permute(0, 2, 1).reshape(B, -1, H, W)  #global_feat: torch.Size([64, 64, 9, 9]) local_feat: torch.Size([64, 64, 9, 9])
        # print("global_feat:", global_feat.shape)
        # print("local_feat:", local_feat.shape)

        #propose
        global2local = torch.sigmoid(local_feat)
        global_feat = global_feat * global2local
        global_feat = self.depthwise(global_feat)
        local_feat = self.pointwise(local_feat)  # global_feat->MFASA; local_feat->GLSL
        x = self.mixer(torch.cat((local_feat, global_feat), dim=1))
        return x

        #消融——注意力机制融合
        # global_weight = self.se(global_feat)
        # local_weight = self.se(local_feat)
        # fused = global_weight + local_weight
        # return fused

        #消add
        # return (local_feat + global_feat)

        #消spe
        # return global_feat

        #消trans
        # return local_feat

        #消cat
        # x = torch.cat((local_feat, global_feat), dim=1)
        # x = x.flatten(2).permute(0, 2, 1)
        # x = self.fc(x)
        # x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        # return x


# model = FASA(128, 8, 11)
# model.eval()
# print(model)
# input = torch.randn(64, 128, 11, 11)
# y = model(input)
# print(y.size())

class FASABlock(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                  ffn=None):
        super().__init__()
        self.ffn = ffn
        self.patch_size = input_resolution[0]
        self.norm1 = norm_layer(dim)
        self.attn = FASA(dim, num_heads, self.patch_size)
        # self.attn = Attention(dim, heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.ffn == 'CGLU':
            self.mlp = ConvolutionalGLU(in_features=dim,
                                        hidden_features=mlp_hidden_dim,
                                        act_layer=act_layer, drop=drop)
        elif self.ffn == 'MLP':
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
        elif self.ffn =='SMF':
            self.mlp = SMFFN(dim, dim*2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = x + self.attn(x)
        x = x.flatten(2).permute(0, 2, 1)
        if self.ffn == 'CGLU':
            x = x + self.drop_path(self.mlp(self.norm2(x), self.patch_size))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x

class CBR_2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias):
        super(CBR_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(self.bn(x))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class TransN(nn.Module):
    def __init__(self, in_channel, num_classes, patch_size):
        super(TransN, self).__init__()
        channels = [128, 64, 32, 256]
        self.in_channel = in_channel
        # self.patch_size = 11
        self.conv1 = CBR_2d(self.in_channel, channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = CBR_2d(channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.fasablc1 = FASABlock(channels[0], 8, to_2tuple(patch_size), ffn='MLP')
        self.fasablc2 = FASABlock(channels[1], 4, to_2tuple(patch_size), ffn='MLP')
        self.s3conv1 = PlainConv(channels[0], channels[0])
        self.s3conv2 = PlainConv(channels[0], channels[1])
        self.pool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(0.)
        self.fc = nn.Linear(channels[1], num_classes, bias=False)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.conv1(x)
        x1 = x.unsqueeze(2)
        x1 = self.s3conv1(x1)
        x1 = x1.squeeze(2)
        x = x + x1

        x = self.fasablc1(x)

        x = self.conv2(x)

        x = self.fasablc2(x)

        x = self.pool(self.dropout(x)).view(-1, x.shape[1])
        x = self.fc(x)
        return x



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model = TransN(200, 16, 11).to(device)
    net.eval()
    print(net)
    input = torch.randn(2, 200, 11, 11).to(device)
    y = net(input)
    print(y.shape, count_parameters(net))