import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
from Model.atten_v6 import Seq2Seq
from Model.Attention import MLCAM
from typing import Optional, Union
from Model.Sample import SafeUpSample, SafeDownSample
import math


class AHDTCN(nn.Module):
    def __init__(self, args, num_f_maps, dim, num_classes):
        super(AHDTCN, self).__init__()
        self.RDs = Reduce_dimension(num_f_maps, dim)
        self.Down = nn.ModuleList([copy.deepcopy(SafeDownSample(channels=num_f_maps)) for _ in range(args.num_hie)])
        self.Up = nn.ModuleList([copy.deepcopy(SafeUpSample(channels=num_f_maps, causal_conv = args.causal_conv)) for _ in range(args.num_hie)])
        self.Conv_res = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_f_maps, 1)) for _ in range(args.num_hie)])
        self.stages1 = nn.ModuleList(
            [copy.deepcopy(HDTCN(args.DifInterval, num_f_maps, args.num_layer, args.causal_conv, args.decomp_flag, args.Difference)) for _ in
             range(args.num_hie)])
        self.stages2 = nn.ModuleList(
            [copy.deepcopy(HDTCN(args.DifInterval, num_f_maps, args.num_layer, args.causal_conv, args.decomp_flag, args.Difference)) for _ in
             range(args.num_hie)])
        self.stage = HDTCN(args.DifInterval, num_f_maps, args.num_layer, args.causal_conv, args.decomp_flag, args.Difference)

        self.Conv_d = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps*2, num_f_maps, 1)) for _ in range(args.num_hie)])

        self.cnn_outs = nn.ModuleList(
            [copy.deepcopy(nn.Conv1d(num_f_maps, num_classes, 1)) for _ in
             range(args.num_hie+1)])
        self.coding_side = args.coding_side
        self.num_hie = args.num_hie

        # self.cnn_outs = nn.Conv1d(num_f_maps, num_classes, 1)
    def forward(self, x):
        out_list = []

        x = self.RDs(x)
        x_d = []

        for i in range(self.num_hie):
            if self.coding_side == "left" or self.coding_side == "both":
                x = self.stages1[i](x)
            x_d.append(x)
            x = self.Down[i](x)
        x = self.stage(x)                         # 中间要有
        x_d.append(x)
        out_list.insert(0, self.cnn_outs[-1](x))

        for j in range(self.num_hie):
            i = self.num_hie - j -1   # j:0,1,2; i:2,1,0
            x_d[i] = self.Conv_d[i](torch.cat([self.Up[i](x_d[i+1], x_d[i].size(2)), self.Conv_res[i](x_d[i])], 1))
            if self.coding_side == "right" or self.coding_side == "both":
                x_d[i] = self.stages2[i](x_d[i])
            out_list.insert(0, self.cnn_outs[i](x_d[i]))

        return out_list

class Up(nn.Module):
    def __init__(self, num_f_maps):
        super(Up, self).__init__()
        self.conv_1x1 = nn.Conv1d(num_f_maps, num_f_maps, 1)

    def forward(self, x):
        x = self.conv_1x1(x)
        return x

class Down(nn.Module):
    def __init__(self, num_f_maps):
        super(Down, self).__init__()
        self.conv_1x1 = nn.Conv1d(num_f_maps, num_f_maps, 1)

    def forward(self, x):
        x = self.conv_1x1(x)
        return x


class HDTCN(nn.Module):
    def __init__(self, DifInterval, num_f_maps, num_layer, causal_conv, decomp_flag=False, Difference=True):
        super(HDTCN, self).__init__()
        # self.dif = [1,4,16,64]
        self.dif = DifInterval
        # self.TCNDs = nn.ModuleList([copy.deepcopy(MulCausalTCN(num_layer, num_f_maps, causal_conv)) for _ in range(len(self.dif))])

        # self.MLCAM = DMLCAM(num_f_maps, 8, 2, 0.1)
        if len(self.dif)>0:
            self.TCND = MulCausalTCN(num_layer, num_f_maps * (len(self.dif) + 1), causal_conv)
            self.MLP = MLP(input_dim=num_f_maps * (len(self.dif) + 1),
                          hidden_dims=num_f_maps * 2,
                          output_dim=num_f_maps * 1,
                          num_layers=1)
        else:
            self.TCN = MulCausalTCN(11, num_f_maps, causal_conv)
            # self.SDP = SecondDimPooling(pool_type='max')
            # self.fc = nn.Linear(num_f_maps * len(self.dif), num_f_maps * 1, bias=False)
            # self.fc1 = nn.Linear(num_f_maps * 2, num_f_maps * 1, bias=False)
        self.num_f_maps = num_f_maps
        self.decomp_flag = decomp_flag
        if self.decomp_flag:
            self.Decomp = series_decomp_multi([2, 4, 8])
        self.Difference = Difference


    def forward(self, x):
        f_ds = []
        if self.decomp_flag:
            _, x_o = self.Decomp(x)
        else:
            x_o = x
        for k in self.dif:
            if self.Difference:
                f = compute_step_diff(x_o, k, self.num_f_maps)
                f_ds.append(f)
            else:
                f_ds.append(x)

        # f_out = self.MLCAM(x)
        if len(self.dif)>0:
            f_ds.append(x)
            fc = torch.cat(f_ds, 1)
            # f_p = self.SDP(f_ds)
            # f_d = self.MLP(fc.transpose(1, 2)).transpose(1, 2) + f_p
            f_d = self.TCND(fc)
            f_out = self.MLP(f_d.transpose(1, 2)).transpose(1, 2)
            # fc = torch.cat([f_d, f_out], 1)
            # f_out = self.fc1(fc.transpose(1,2)).transpose(1,2)
        else:
            f_out = self.TCN(x)

        return f_out


class SecondDimPooling(nn.Module):
    def __init__(self, pool_type='max'):
        super(SecondDimPooling, self).__init__()
        self.pool_type = pool_type

        if pool_type not in ['max', 'avg', 'sum']:
            raise ValueError("pool_type 必须是 'max', 'avg' 或 'sum'")

    def forward(self, tensor_list):
        if not tensor_list:
            raise ValueError("输入列表不能为空")

        # 检查所有张量形状是否一致
        first_shape = tensor_list[0].shape
        for tensor in tensor_list:
            assert tensor.shape == first_shape, "所有张量形状必须相同"
            assert tensor.shape[0] == 1 and tensor.shape[1] == 64, "张量形状必须为 [1,64,T]"

        # 沿新维度堆叠张量 [N,1,64,T], N是列表长度
        stacked = torch.stack(tensor_list, dim=0)

        if self.pool_type == 'max':
            # 最大池化
            pooled, _ = torch.max(stacked, dim=0)
        elif self.pool_type == 'avg':
            # 平均池化
            pooled = torch.mean(stacked, dim=0)
        elif self.pool_type == 'sum':
            # 求和池化
            pooled = torch.sum(stacked, dim=0)

        return pooled  # 形状 [1,64,T]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=1, dropout=0.1):
        """
        Multi-Layer Perceptron (MLP) module

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions or a single integer for all hidden layers
            output_dim: Dimension of output features
            num_layers: Total number of layers (including input and output)
            dropout: Dropout probability
        """
        super(MLP, self).__init__()

        # If hidden_dims is an integer, make all hidden layers the same size
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * (num_layers - 2)

        # Build the layer list
        layers = []
        in_dim = input_dim

        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim, bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Add output layer (without activation)
        layers.append(nn.Linear(in_dim, output_dim, bias=True))

        # Combine all layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # x = x.transpose(1,2)
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        # x = x.transpose(1, 2)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        x0 = x
        x = x0.transpose(1, 2)  ##
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        moving_mean = moving_mean.transpose(1, 2)  ##
        res = x0 - moving_mean
        return res, moving_mean

def compute_step_diff(x, K, num_f_maps, pad_value=0.0):
    """
    计算序列的 K 步差分，并保持输出形状不变（前面填充 pad_value）

    Args:
        x: 输入序列，形状 [1, 64, T]
        K: 步长（每个值减去前 K 个位置的值）
        pad_value: 填充值（默认 0）

    Returns:
        diff: 形状 [1, 64, T]，前 K 个位置用 pad_value 填充
    """
    if K == 0:
        return x.clone()  # 如果 K=0，直接返回原序列

    # 计算差分：x[:, :, K:] - x[:, :, :-K]
    diff = x[:, :, K:] - x[:, :, :-K]

    # 在前面填充 K 个 pad_value
    padding = torch.full((1, num_f_maps, K), pad_value, dtype=x.dtype, device=x.device)
    diff_padded = torch.cat([padding, diff], dim=2)[:,:,:x.size(2)]

    return diff_padded




class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)



class Reduce_dimension(nn.Module):
    def __init__(self, num_f_maps, dim):
        super(Reduce_dimension, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.channel_dropout = nn.Dropout2d()


    def forward(self, x):
        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)
        x = self.conv_1x1(x)
        return x



class MulCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, causal_conv = True):
        super(MulCausalTCN, self).__init__()
        # self.layers = nn.ModuleList(
        #     [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])

        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps,num_f_maps,
                causal_conv=causal_conv))
            for i in range(num_layers)
        ])

        self.channel_dropout = nn.Dropout2d()
        self.num_layers = num_layers
    def forward(self, x):
        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)
        for layer in self.layers:
            x = layer(x)
        # for i in range(self.num_layers):
        #     x = self.layers[i](x)
        return x


# class DMLCAM(nn.Module):
#     def __init__(self, num_f_maps, head, layer, dropout):
#         super(DMLCAM, self).__init__()
#         self.MLCAM = MLCAM(num_f_maps, head, layer, dropout)
#         # self.channel_dropout = nn.Dropout2d()
#         self.dropout = nn.Dropout1d(p=0.1)
#     def forward(self, x):
#         # x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
#         # x = self.channel_dropout(x)
#         # x = x.squeeze(3)
#         # x = random_zero_mask(x, mode="independent", drop_prob=0.3)
#         x = self.dropout(x)
#         x = self.MLCAM(x.permute(2, 0, 1), x.permute(2, 0, 1), x.permute(2, 0, 1)).permute(1, 2, 0) + x
#         return x

class DMLCAM(nn.Module):
    def __init__(self, num_f_maps, head, layer, dropout, num = 8):
        super(DMLCAM, self).__init__()
        self.MLCAM = MLCAM(num_f_maps, head, layer, dropout)
        # self.channel_dropout = nn.Dropout2d()
        self.dropout = nn.Dropout1d(p=0.1)
        self.num = num
    def forward(self, x):
        x = self.dropout(x)
        for i in range(self.num):
            x_ = x[:,:,i::self.num]
            x[:,:,i::self.num] = self.MLCAM(x_.permute(2, 0, 1), x_.permute(2, 0, 1), x_.permute(2, 0, 1)).permute(1, 2, 0) + x_
        # x = self.MLCAM(x.permute(2, 0, 1), x.permute(2, 0, 1), x.permute(2, 0, 1)).permute(1, 2, 0) + x
        return x


def random_zero_mask(
        x: torch.Tensor,
        mode: str = "independent",
        drop_prob: float = 0.3,
        max_drop_length: Optional[int] = None,
        drop_ratio: Optional[float] = None,
) -> torch.Tensor:
    """
    对输入张量沿时间维度 (T) 随机置 0

    Args:
        x: 输入张量，形状为 [..., C, T]（支持任意前导维度）
        mode: 置 0 模式，可选 "independent"（独立随机）/ "block"（整段）/ "ratio"（固定比例）
        drop_prob: 独立随机置 0 的概率（mode="independent" 时生效）
        max_drop_length: 整段置 0 时的最大连续长度（mode="block" 时生效）
        drop_ratio: 固定置 0 的比例（mode="ratio" 时生效）

    Returns:
        torch.Tensor: 部分时间步被置 0 的张量，形状与输入相同
    """
    assert len(x.shape) >= 2, "输入张量至少需要包含 C 和 T 维度"
    assert mode in ["independent", "block", "ratio"], "模式必须是 independent/block/ratio"

    # 统一处理成 [..., 1, T] 的掩码以便广播
    original_shape = x.shape
    C, T = x.shape[-2], x.shape[-1]
    x_reshaped = x.reshape(-1, 1, T)  # [..., 1, T]

    if mode == "independent":
        mask = (torch.rand_like(x_reshaped) > drop_prob).float()
    elif mode == "block":
        assert max_drop_length is not None, "整段模式需指定 max_drop_length"
        mask = torch.ones_like(x_reshaped)
        drop_start = torch.randint(0, T - max_drop_length, (1,)).item()
        drop_end = drop_start + torch.randint(1, max_drop_length + 1, (1,)).item()
        mask[..., drop_start:drop_end] = 0
    elif mode == "ratio":
        assert drop_ratio is not None, "固定比例模式需指定 drop_ratio"
        num_to_drop = int(T * drop_ratio)
        drop_indices = torch.randperm(T)[:num_to_drop]
        mask = torch.ones_like(x_reshaped)
        mask[..., drop_indices] = 0

    # 应用掩码并恢复原始形状
    masked_x = x_reshaped * mask
    return masked_x.reshape(original_shape)

