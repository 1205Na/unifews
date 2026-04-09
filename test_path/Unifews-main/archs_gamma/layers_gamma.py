import os
os.environ['TL_BACKEND'] = 'torch'

from math import log
import numpy as np
from typing import Optional, Tuple, Union, Any

# ===================== 核心替换：PyTorch/PyG → TLX/GammaGL =====================
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn import Linear
import gammagl as ggl

# 【修复】TLX 没有独立的 Tensor 类，使用 Any 作为类型标注
Tensor = Any
Adj = Tensor
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, OptTensor]

from gammagl.layers.conv import (
    GCNConv, GATV2Conv, GINConv, SAGEConv, MessagePassing
)
GATv2Conv = GATV2Conv

# 重写后的剪枝工具
from .prunes_gamma import ThrInPrune, rewind,prune
from utils.logger_gamma import LayerNumLogger

# ==============================================
# ===== 【核心新增】tlx版 PyG 标准初始化函数 =====
# ==============================================
def norm(x, p=2, axis=None, keepdims=False):
    """
    Manual implementation of the norm function for TensorLayerX.
    """
    # Ensure we are working with absolute values for the norm calculation
    x_abs = tlx.abs(x)
    
    if p == 1:
        return tlx.reduce_sum(x_abs, axis=axis, keepdims=keepdims)
    elif p == 2:
        # Standard Euclidean norm: sqrt(sum(x^2))
        return tlx.sqrt(tlx.reduce_sum(x_abs * x_abs, axis=axis, keepdims=keepdims))
    else:
        # General Lp norm
        sum_p = tlx.reduce_sum(tlx.pow(x_abs, p), axis=axis, keepdims=keepdims)
        return tlx.pow(sum_p, 1.0/p)

def normalize(x, p=2., axis=-1):
    # Use our new manual norm function here
    norm_val = norm(x, p=p, axis=axis, keepdims=True)
    return x / (norm_val + 1e-12)


def reset_bn_(bn_module):
    """
    Manually resets BatchNorm parameters: 
    gamma -> 1.0, beta -> 0.0, moving_mean -> 0.0, moving_var -> 1.0
    """
    if bn_module is None:
        return

    # 1. Reset Scale (Gamma) to 1.0
    if hasattr(bn_module, 'gamma') and bn_module.gamma is not None:
        new_gamma = tlx.initializers.Ones()(bn_module.gamma.shape)
        if tlx.BACKEND == 'torch':
            import torch
            with torch.no_grad():
                bn_module.gamma.data.copy_(torch.as_tensor(new_gamma, dtype=bn_module.gamma.dtype))
        else:
            bn_module.gamma.assign(new_gamma)

    # 2. Reset Shift (Beta) to 0.0
    if hasattr(bn_module, 'beta') and bn_module.beta is not None:
        new_beta = tlx.initializers.Zeros()(bn_module.beta.shape)
        if tlx.BACKEND == 'torch':
            import torch
            with torch.no_grad():
                bn_module.beta.data.copy_(torch.as_tensor(new_beta, dtype=bn_module.beta.dtype))
        else:
            bn_module.beta.assign(new_beta)

    # 3. Reset Running Mean to 0.0
    if hasattr(bn_module, 'moving_mean') and bn_module.moving_mean is not None:
        new_mean = tlx.initializers.Zeros()(bn_module.moving_mean.shape)
        if tlx.BACKEND == 'torch':
            import torch
            with torch.no_grad():
                bn_module.moving_mean.data.copy_(torch.as_tensor(new_mean, dtype=bn_module.moving_mean.dtype))
        else:
            bn_module.moving_mean.assign(new_mean)

    # 4. Reset Running Variance to 1.0
    if hasattr(bn_module, 'moving_var') and bn_module.moving_var is not None:
        new_var = tlx.initializers.Ones()(bn_module.moving_var.shape)
        if tlx.BACKEND == 'torch':
            import torch
            with torch.no_grad():
                bn_module.moving_var.data.copy_(torch.as_tensor(new_var, dtype=bn_module.moving_var.dtype))
        else:
            bn_module.moving_var.assign(new_var)

def reset_weight_(weight: Tensor, in_channels: int, initializer: Optional[str] = None) -> Tensor:
    if weight is None:
        return weight
    shape = weight.shape
    device = weight.device if hasattr(weight, 'device') else None
    if in_channels <= 0:
        pass
    elif initializer == 'glorot':
        new_weight = tlx.initializers.XavierUniform()(shape)
    elif initializer == 'uniform':
        bound = 1.0 / np.sqrt(in_channels)
        new_weight = tlx.initializers.RandomUniform(-bound, bound)(shape)
    elif initializer == 'kaiming_uniform' or initializer is None:
        new_weight = tlx.initializers.HeUniform()(shape)
    else:
        raise RuntimeError(f"Weight initializer '{initializer}' not supported")
    if device is not None:
        new_weight = tlx.convert_to_tensor(new_weight, device=device)
    if hasattr(weight, 'data'):
        weight.data = new_weight
    else:
        weight.copy_(new_weight)
    return weight

def reset_bias_(bias: Optional[Tensor], in_channels: int, initializer: Optional[str] = None) -> Optional[Tensor]:
    if bias is None or in_channels <= 0:
        return bias
    shape = bias.shape
    device = bias.device if hasattr(bias, 'device') else None
    if initializer == 'zeros':
        new_bias = tlx.initializers.Zeros()(shape)
    elif initializer == 'uniform' or initializer is None:
        bound = 1.0 / np.sqrt(in_channels)
        new_bias = tlx.initializers.RandomUniform(-bound, bound)(shape)
    else:
        raise RuntimeError(f"Bias initializer '{initializer}' not supported")
    if device is not None:
        new_bias = tlx.convert_to_tensor(new_bias, device=device)
    if hasattr(bias, 'data'):
        bias.data = new_bias
    else:
        bias.copy_(new_bias)
    return bias

# ===================== 独立功能函数实现 =====================
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    return int(tlx.reduce_max(edge_index)) + 1

def scatter(src, index, axis=0, dim_size=None, reduce='sum'):
    if dim_size is None:
        dim_size = int(tlx.reduce_max(index)) + 1
    if reduce == 'sum':
        return tlx.unsorted_segment_sum(src, index, num_segments=dim_size)
    elif reduce == 'max':
        return tlx.unsorted_segment_max(src, index, num_segments=dim_size)
    elif reduce == 'mean':
        return tlx.unsorted_segment_mean(src, index, num_segments=dim_size)
    elif reduce == 'min':
        return tlx.unsorted_segment_min(src, index, num_segments=dim_size)
    else:
        raise ValueError(f"Unsupported reduce type: {reduce}")

def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1.0, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    
    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    edge_index = edge_index[:, mask]
    
    loop_index = tlx.convert_to_tensor(np.arange(num_nodes), dtype=edge_index.dtype)
    loop_index = tlx.stack([loop_index, loop_index], axis=0)
    
    edge_index = tlx.concat([edge_index, loop_index], axis=1)
    
    if edge_weight is not None:
        loop_weight = tlx.ones((num_nodes,), dtype=edge_weight.dtype) * fill_value
        edge_weight = tlx.concat([edge_weight, loop_weight], axis=0)
        
    return edge_index, edge_weight

def pow_with_pinv(x, p: float):
    x = tlx.convert_to_tensor(x)
    x_pow = tlx.pow(x, p)
    x_safe = tlx.where(tlx.is_inf(x_pow), tlx.zeros_like(x_pow), x_pow)
    return x_safe

def leaky_relu(x, negative_slope=0.2):
    return tlx.where(x > 0.0, x, x * negative_slope)

def normalize(x, p=2., axis=-1):
    norm_val = tlx.norm(x, axis=axis, keepdims=True)
    return x / (norm_val + 1e-12)

def softmax(src, index, ptr=None, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(tlx.reduce_max(index)) + 1
    src_max = scatter(src, index, axis=0, dim_size=num_nodes, reduce='max')
    src_max_gathered = tlx.gather(src_max, index)
    out = tlx.exp(src - src_max_gathered)
    out_sum = scatter(out, index, axis=0, dim_size=num_nodes, reduce='sum')
    out_sum_gathered = tlx.gather(out_sum, index)
    return out / (out_sum_gathered + 1e-16)

def gcn_norm(edge_index, edge_weight, num_nodes, improved=False, add_self_loops=True, flow="source_to_target", dtype=tlx.float32):
    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1.0, num_nodes)
    if edge_weight is None:
        edge_weight = tlx.ones((edge_index.shape[1],), dtype=dtype)
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, axis=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = pow_with_pinv(deg, -0.5)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight

class MLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lins = nn.ModuleList([
            Linear(in_channels=channels[i], out_channels=channels[i+1]) 
            for i in range(len(channels)-1)
        ])
        self.reset_parameters() # ===== FIXED：MLP初始化 =====

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = tlx.relu(x)
        return x

    def reset_parameters(self):
        for lin in self.lins:
            reset_weight_(lin.weights, lin.in_channels, initializer='kaiming_uniform')
            reset_bias_(lin.biases, lin.in_channels, initializer='uniform')

# ===================== 图归一化函数 =====================
def identity_n_norm(edge_index, edge_weight=None, num_nodes=None,
                    rnorm=None, diag=1., dtype=tlx.float32):
    if tlx.is_tensor(edge_index):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if diag is not None:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, diag, num_nodes)
        if rnorm is None:
            return edge_index
        else:
            edge_weight = tlx.ones((edge_index.shape[1], ), dtype=dtype,
                                     device=edge_index.device)
            row, col = edge_index[0], edge_index[1]
            idx = col
            deg = scatter(edge_weight, idx, axis=0, dim_size=num_nodes, reduce='sum')
            deg_inv_sqrt = pow_with_pinv(deg, -0.5)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight
    raise NotImplementedError()

# ===================== GCNII 卷积层 =====================
class GCNIIConv(MessagePassing):
    _cached_edge_index = None
    _cached_adj_t = None

    def __init__(self, channels: int, channels_fake: int, alpha: float, theta: float = None,
                 depth: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or depth is not None:
            assert theta is not None and depth is not None
            self.beta = log(theta / (depth + 1) + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin1 = Linear(channels, channels, bias=False)
        self.lin2 = Linear(channels, channels, bias=False) if not shared_weights else None
        self.reset_parameters()

    # ===== 已经正确，无需修改 =====
    def reset_parameters(self):
        reset_weight_(self.lin1.weights, self.channels,initializer='glorot') # ===== FIXED：用tlx初始化 =====
        if self.lin2 is not None:
            reset_weight_(self.lin2.weights, self.channels,initializer='glorot')
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, x_0, edge_index: Adj, edge_weight: OptTensor = None):
        if self.normalize:
            if tlx.is_tensor(edge_index):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.shape[getattr(self, 'node_dim', 0)], False,
                        self.add_self_loops, getattr(self, 'flow', 'source_to_target'), dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x = x * (1 - self.alpha)
        x_0 = self.alpha * x_0[:x.shape[0]]
        if self.lin2 is None:
            out = x + x_0
            out = out * (1. - self.beta) + tlx.matmul(out, self.lin1.weight) * self.beta
        else:
            out = x * (1. - self.beta) + tlx.matmul(x, self.lin1.weight) * self.beta
            out = out + x_0 * (1. - self.beta) + tlx.matmul(x_0, self.lin2.weight) * self.beta
        return out

    def message(self, x_j, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, alpha={self.alpha}, beta={self.beta})'

# ===================== 剪枝基类 ConvThr =====================
class ConvThr(nn.Module):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_a = thr_a
        self.threshold_w = thr_w
        self.idx_keep = tlx.convert_to_tensor([])
        self.prune_lst = []
        self.scheme_a = 'full'
        self.scheme_w = 'full'

    def propagate_forward_print(self, module, inputs, output):
        print(inputs[0], inputs[0].shape, inputs[1])
        print(output, output.shape)

    def get_idx_lock(self, edge_index, node_lock):
        idx_lock = tlx.convert_to_tensor([], dtype=tlx.int32, device=edge_index.device)
        bs = int(2**28 / edge_index.shape[1])
        for i in range(0, node_lock.shape[0], bs):
            batch = node_lock[i:min(i+bs, node_lock.shape[0])]
            idx_lock = tlx.concat((idx_lock, tlx.where(edge_index[1].unsqueeze(0) == batch.unsqueeze(1))[1]))
        idx_diag = tlx.where(edge_index[0] == edge_index[1])[0]
        idx_lock = tlx.concat((idx_lock, idx_diag))
        return tlx.unique(idx_lock)

# ===================== GCN 系列 =====================
class GCNConvRaw(GCNConv):
    def __init__(self, in_channels, out_channels, 
                 rnorm=None, diag=1., depth_inv=False, *args, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters() # ===== FIXED：构造时调用 =====

    # ==============================================
    # ===== FIXED：GCNConv 完整 reset_parameters =====
    # ==============================================
    def reset_parameters(self):
        # 1. 初始化线性层（权重+偏置）
        reset_weight_(self.linear.weights, self.in_channels,initializer='kaiming_uniform')
        reset_bias_(self.linear.biases, self.in_channels,initializer='zeros')
        # 2. 清空缓存
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_tuple: PairTensor, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.lin.weight.numel()
        return super().forward(x, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, (edge_index, edge_weight) = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        flops_bias = f_out if module.lin.bias is not None else 0
        module.__flops__ += int(f_in * f_out * n)
        module.__flops__ += flops_bias * n
        module.__flops__ += f_in * m

class GCNConvRnd(ConvThr, GCNConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin]
        self.idx_keep = None

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if prune.is_pruned(self.lin):
                prune.remove(self.lin, 'weight')
            if self.scheme_w == 'pruneall':
                amount = self.threshold_w
            else:
                amount = int(self.lin.weight.numel() * (1-self.threshold_w)) - tlx.sum(self.lin.weight == 0).item()
                amount = max(amount, 0)
            prune.RandomUnstructured.apply(self.lin, 'weight', amount)
            x = self.lin(x)
        elif self.scheme_w == 'keep':
            x = self.lin(x)
        elif self.scheme_w == 'full':
            raise NotImplementedError()
        self.logger_w.numel_before = self.lin.weight.numel()
        self.logger_w.numel_after = tlx.sum(self.lin.weight != 0).item()

        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            if self.idx_keep is None:
                self.idx_keep = tlx.randperm(edge_index.shape[1])[:int(edge_index.shape[1]*(1-self.threshold_a))]
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]
            edge_index = edge_index[:, self.idx_keep]
            edge_weight = edge_weight[self.idx_keep]
        else:
            self.logger_a.numel_after = edge_index.shape[1]
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias is not None: out = out + self.bias
        return out, (edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, _ = input
        x_out, (edge_index, edge_weight) = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        flops_bias = f_out if module.lin.bias is not None else 0
        module.__flops__ += int((f_in * f_out * module.logger_w.ratio + flops_bias) * n)
        module.__flops__ += f_in * m

class GCNConvThr(ConvThr, GCNConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.bias]
        self.register_forward_hook(self.prune_on_msg)

    def prune_on_msg(self, module, inputs, output):
        if self.scheme_a in ['pruneall', 'pruneinc']:
            mask_0 = tlx.zeros(output.shape[0], dtype=tlx.bool, device=output.device)
            norm_feat_msg = tlx.norm(output, axis=1)
            norm_all_msg = tlx.norm(norm_feat_msg, axis=None, p=1) / output.shape[0]
            mask_cmp = norm_feat_msg < self.threshold_a * norm_all_msg
            mask_0 = tlx.logical_or(mask_0, mask_cmp)
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
            self.idx_keep = tlx.where(~mask_0)[0]
        elif self.scheme_a == 'keep':
            mask_0 = tlx.ones(output.shape[0], dtype=tlx.bool)
            mask_0[self.idx_keep] = False
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
        return output

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.linear): rewind(self.linear, 'weight')
            else:
                if prune.is_pruned(self.linear): prune.remove(self.linear, 'weight')
            norm_node_in = norm(x, axis=0)
            norm_all_in = norm(norm_node_in, axis=None)/x.shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                ThrInPrune.apply(self.linear, 'weights', threshold_wi)
            x = self.lin(x)
        elif self.scheme_w == 'keep':
            x = self.lin(x)
        elif self.scheme_w == 'full':
            raise NotImplementedError()
        self.logger_w.numel_before = self.lin.weight.numel()
        self.logger_w.numel_after = tlx.sum(self.lin.weight != 0).item()
        self.idx_lock = self.get_idx_lock(edge_index, node_lock)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias is not None: out = out + self.bias
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]
            edge_index = edge_index[:, self.idx_keep]
            edge_weight = edge_weight[self.idx_keep]
        return out, (edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, _ = input
        x_out, (edge_index, edge_weight) = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        flops_bias = f_out if module.lin.bias is not None else 0
        module.__flops__ += int((f_in * f_out * module.logger_w.ratio + flops_bias) * n)
        module.__flops__ += f_in * (m - n)

# ===================== GATv2 系列 =====================
class GATv2ConvRaw(GATV2Conv):
    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 rnorm=None, diag=1., depth_inv=False, heads: int = 1, concat: bool = True, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        heads = 1 if depth == 0 else heads
        concat = (depth > 0)
        if concat: out_channels = out_channels // heads
        super().__init__(in_channels, out_channels, heads, concat,** kwargs)
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters() # ===== FIXED：构造时调用 =====

    # ==============================================
    # ===== FIXED：GATv2Conv 完整 reset_parameters =====
    # ==============================================
    def reset_parameters(self):
    # 权重：glorot | 注意力参数：glorot | 偏置：zeros
        reset_weight_(self.lin_l.weights, self.in_channels, initializer='glorot')
        if not self.share_weights:
            reset_weight_(self.lin_r.weights, self.in_channels, initializer='glorot')
    # 注意力向量初始化
        reset_weight_(self.att, self.out_channels, initializer='glorot')
    # 偏置全0（GAT系列强制规范）
        reset_bias_(self.biases, self.heads * self.out_channels, initializer='zeros')
    # 清空缓存
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index: Adj, edge_weight: OptTensor = None, **kwargs):
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.lin_l.weight.numel()
        if not self.share_weights: self.logger_w.numel_after += self.lin_r.weight.numel()
        return super().forward(x, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_index = input
        f_in, f_h, f_c = x_in.shape[-1], module.heads, module.out_channels
        n, m = x_in.shape[0], edge_index.shape[1]
        flops_lin = f_in * f_h * f_c * n
        if not module.share_weights: flops_lin *= 2
        module.__flops__ += flops_lin
        flops_attn  = (2 * f_c + 2) * m * f_h
        module.__flops__ += flops_attn
        if module.bias is not None: module.__flops__ += (f_h * f_c if module.concat else f_c + 1) * n

class GATv2ConvRnd(ConvThr, GATv2ConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin_l, self.lin_r]
        self.idx_keep = None

    def forward(self, x, edge_index: Adj, edge_attr: OptTensor = None, node_lock: OptTensor = None, verbose: bool = False):
        H, C = self.heads, self.out_channels
        assert x.dim() == 2
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if prune.is_pruned(self.lin_l):
                prune.remove(self.lin_l, 'weight')
                prune.remove(self.lin_r, 'weight')
            linset = (self.lin_l,) if self.share_weights else (self.lin_l, self.lin_r)
            for lin in linset:
                amount = self.threshold_w if self.scheme_w=='pruneall' else max(int(lin.weight.numel()*(1-self.threshold_w))-tlx.sum(lin.weight==0).item(),0)
                prune.RandomUnstructured.apply(lin, 'weight', amount)
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)
        elif self.scheme_w == 'keep':
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)
        else: raise NotImplementedError()
        self.logger_w.numel_before = self.lin_l.weight.numel() + (self.lin_r.weight.numel() if not self.share_weights else 0)
        self.logger_w.numel_after = tlx.sum(self.lin_l.weight!=0) + (tlx.sum(self.lin_r.weight!=0) if not self.share_weights else 0)
        if self.scheme_a in ['pruneall','pruneinc','keep']:
            if self.idx_keep is None: self.idx_keep = tlx.randperm(edge_index.shape[1])[:int(edge_index.shape[1]*(1-self.threshold_a))]
            edge_index = edge_index[:, self.idx_keep]
            if edge_attr is not None: edge_attr = edge_attr[self.idx_keep]
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, size=None)
        out = tlx.reshape(out, (-1, self.heads*self.out_channels)) if self.concat else tlx.reduce_mean(out, axis=1)
        if self.bias is not None: out += self.bias
        return out, edge_index

    def edge_update(self, x_j, x_i, edge_attr: OptTensor, index, ptr: OptTensor, dim_size: Optional[int]):
        x = x_i + x_j
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr.view(-1,1) if edge_attr.dim()==1 else edge_attr).view(-1, self.heads, self.out_channels)
            x += edge_attr
        x = leaky_relu(x, self.negative_slope)
        alpha = tlx.reduce_sum(x * self.att, axis=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        return tlx.dropout(alpha, p=self.dropout, training=self.training)

class GATv2ConvThr(ConvThr, GATv2ConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin_l, self.lin_r]

    def forward(self, x, edge_index: Adj, edge_attr: OptTensor = None, node_lock: OptTensor = None, verbose: bool = False):
        H, C = self.heads, self.out_channels
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.lin_l): rewind(self.lin_l,'weight'),rewind(self.lin_r,'weight')
            else:
                if prune.is_pruned(self.lin_l): prune.remove(self.lin_l,'weight'),prune.remove(self.lin_r,'weight')
            norm_node_in = tlx.norm(x, axis=0)
            norm_all_in = tlx.norm(norm_node_in, axis=None)/x.shape[1]
            if norm_all_in>1e-8:
                threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                ThrInPrune.apply(self.lin_l, 'weight', threshold_wi)
                if not self.share_weights: ThrInPrune.apply(self.lin_r, 'weight', threshold_wi)
            x_l = self.lin_l(x).view(-1,H,C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1,H,C)
        elif self.scheme_w == 'keep':
            x_l = self.lin_l(x).view(-1,H,C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1,H,C)
        else: raise NotImplementedError()
        self.idx_lock = self.get_idx_lock(edge_index, node_lock)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, size=None)
        out = tlx.reshape(out, (-1, self.heads*self.out_channels)) if self.concat else tlx.reduce_mean(out, axis=1)
        if self.bias is not None: out += self.bias
        if self.scheme_a in ['pruneall','pruneinc','keep']:
            edge_index = edge_index[:, self.idx_keep]
            if edge_attr is not None: edge_attr = edge_attr[self.idx_keep]
        return out, edge_index

    def edge_update(self, x_j, x_i, edge_attr: OptTensor, index, ptr: OptTensor, size_i: Optional[int]):
        x = x_i + x_j
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr.view(-1,1) if edge_attr.dim()==1 else edge_attr).view(-1, self.heads, self.out_channels)
            x += edge_attr
        x = leaky_relu(x, self.negative_slope)
        alpha = tlx.reduce_sum(x * self.att, axis=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        if self.scheme_a in ['pruneall','pruneinc']:
            mask_0 = tlx.zeros(alpha.shape[0], dtype=tlx.bool, device=alpha.device)
            norm_feat_msg = tlx.norm(x_j, axis=[1,2])
            norm_all_msg = tlx.norm(norm_feat_msg, axis=None, p=1)/x_j.shape[0]
            threshold_aj = self.threshold_a * norm_all_msg / norm_feat_msg
            mask_cmp = (tlx.norm(alpha, axis=1)/alpha.shape[1]) < threshold_aj
            mask_0 = tlx.logical_or(mask_0, mask_cmp)
            mask_0[self.idx_lock] = False
            alpha[mask_0] = 0
            self.idx_keep = tlx.where(~mask_0)[0]
        elif self.scheme_a == 'keep':
            mask_0 = tlx.ones(alpha.shape[0], dtype=tlx.bool)
            mask_0[self.idx_keep] = False
            mask_0[self.idx_lock] = False
            alpha[mask_0] = 0
        return tlx.dropout(alpha, p=self.dropout, training=self.training)

# ===================== GIN / GCNII / SAGE 系列 =====================
class GINConvRaw(GINConv):
    def __init__(self, in_channels: int, out_channels: int,
                 rnorm=None, diag=1., depth_inv=False, eps: float = 0., train_eps: bool = False,** kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        nn_default = MLP([in_channels, out_channels])
        super().__init__(nn_default, eps, train_eps, **kwargs)
        self.reset_parameters() # ===== FIXED：构造时调用 =====

    # ==============================================
    # ===== FIXED：GINConv 完整 reset_parameters =====
    # ==============================================
    def reset_parameters(self):
        self.nn.reset_parameters() # MLP已实现初始化

class GCNIIConvRaw(GCNIIConv):
    def __init__(self, *args, rnorm=None, diag=1., depth_inv=False,** kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        super().__init__(*args, **kwargs)
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()


    def forward(self, x, x_0, edge_tuple: PairTensor):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.lin1.weight.numel() + (self.lin2.weight.numel() if self.lin2 else 0)
        return super().forward(x, x_0, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, x_0, (edge_index, edge_weight) = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        module.__flops__ += int(f_in * f_out * n) * (2 if module.lin2 else 1)
        module.__flops__ += f_in * m

class GCNIIConvThr(ConvThr, GCNIIConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin1]
        if self.lin2 is not None: self.prune_lst.append(self.lin2)
        self.register_message_forward_hook(self.prune_on_msg)

    def prune_on_msg(self, module, inputs, output):
        if self.scheme_a in ['pruneall','pruneinc']:
            mask_0 = tlx.zeros(output.shape[0], dtype=tlx.bool, device=output.device)
            norm_feat_msg = tlx.norm(output, axis=1)
            norm_all_msg = tlx.norm(norm_feat_msg, axis=None, p=1)/output.shape[0]
            mask_cmp = norm_feat_msg < self.threshold_a * norm_all_msg
            mask_0 = tlx.logical_or(mask_0, mask_cmp)
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
            self.idx_keep = tlx.where(~mask_0)[0]
        elif self.scheme_a == 'keep':
            mask_0 = tlx.ones(output.shape[0], dtype=tlx.bool)
            mask_0[self.idx_keep] = False
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
        return output

    def forward(self, x, x_0, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        def trans(xx, xx_0):
            if self.lin2 is None:
                out = xx + xx_0
                return out*(1-self.beta) + tlx.matmul(out, self.lin1.weight)*self.beta
            else:
                return xx*(1-self.beta)+tlx.matmul(xx,self.lin1.weight)*self.beta + xx_0*(1-self.beta)+tlx.matmul(xx_0,self.lin2.weight)*self.beta
        (edge_index, edge_weight) = edge_tuple
        self.idx_lock = self.get_idx_lock(edge_index, node_lock)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        x = x*(1-self.alpha)
        x_0 = self.alpha * x_0[:x.shape[0]]
        if self.scheme_w in ['pruneall','pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.lin1): rewind(self.lin1,'weight')
                if self.lin2 and prune.is_pruned(self.lin2): rewind(self.lin2,'weight')
            else:
                if prune.is_pruned(self.lin1): prune.remove(self.lin1,'weight')
                if self.lin2 and prune.is_pruned(self.lin2): prune.remove(self.lin2,'weight')
            norm_node_in = tlx.norm(x, axis=0)
            norm_all_in = tlx.norm(norm_node_in, axis=None)/x.shape[1]
            if norm_all_in>1e-8:
                threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                ThrInPrune.apply(self.lin1, 'weight', threshold_wi)
                if self.lin2: ThrInPrune.apply(self.lin2, 'weight', threshold_wi)
            out = trans(x,x_0)
        elif self.scheme_w == 'keep':
            out = trans(x,x_0)
        else: raise NotImplementedError()
        return out, (edge_index[:,self.idx_keep], edge_weight[self.idx_keep])

class SAGEConvRaw(SAGEConv):
    def __init__(self, in_channels, out_channels, rnorm=None, diag=1., depth_inv=False, *args, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters() # ===== FIXED：构造时调用 =====

    # ==============================================
    # ===== FIXED：SAGEConv 完整 reset_parameters =====
    # ==============================================
    def reset_parameters(self):
    # 邻居线性层：权重kaiming_uniform | 偏置uniform
        reset_weight_(self.lin_l.weights, self.in_channels, initializer='kaiming_uniform')
        reset_bias_(self.lin_l.biases, self.in_channels, initializer='uniform')
    # 自身线性层（如有）
        if self.root_weight:
            reset_weight_(self.lin_r.weights, self.in_channels, initializer='kaiming_uniform')

    def forward(self, x, edge_tuple: PairTensor, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.lin_l.weight.numel() + (self.lin_r.weight.numel() if self.root_weight else 0)
        return super().forward(x, edge_index, size=None)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_tuple = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_tuple[0].shape[1]
        module.__flops__ += int(f_in * f_out * n) * (2 if module.root_weight else 1)
        module.__flops__ += (f_out if module.lin_l.bias else 0) * n
        module.__flops__ += f_in * m

class SAGEConvThr(ConvThr, SAGEConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin_l]
        if self.root_weight: self.prune_lst.append(self.lin_r)
        self.register_message_forward_hook(self.prune_on_msg)

    def prune_on_msg(self, module, inputs, output):
        if self.scheme_a in ['pruneall','pruneinc']:
            mask_0 = tlx.zeros(output.shape[0], dtype=tlx.bool, device=output.device)
            norm_feat_msg = tlx.norm(output, axis=1)
            norm_all_msg = tlx.norm(norm_feat_msg, axis=None, p=1)/output.shape[0]
            mask_cmp = norm_feat_msg < self.threshold_a * norm_all_msg
            mask_0 = tlx.logical_or(mask_0, mask_cmp)
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
            self.idx_keep = tlx.where(~mask_0)[0]
        elif self.scheme_a == 'keep':
            mask_0 = tlx.ones(output.shape[0], dtype=tlx.bool)
            mask_0[self.idx_keep] = False
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
        return output

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        def prune_w(lin, xx):
            if self.scheme_w in ['pruneall','pruneinc']:
                if self.scheme_w == 'pruneall':
                    if prune.is_pruned(lin): rewind(lin,'weight')
                else:
                    if prune.is_pruned(lin): prune.remove(lin,'weight')
                norm_node_in = tlx.norm(xx, axis=0)
                norm_all_in = tlx.norm(norm_node_in, axis=None)/xx.shape[1]
                if norm_all_in>1e-8:
                    threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                    ThrInPrune.apply(lin, 'weight', threshold_wi)
                return lin(xx)
            return lin(xx) if self.scheme_w=='keep' else None
        (edge_index, edge_weight) = edge_tuple
        x = (x, x) if tlx.is_tensor(x) else x
        self.idx_lock = self.get_idx_lock(edge_index, node_lock)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = prune_w(self.lin_l, out)
        if self.root_weight: out += prune_w(self.lin_r, x[1])
        if self.normalize: out = normalize(out)
        return out, (edge_index[:,self.idx_keep], edge_weight[self.idx_keep])

# ===================== FLOPs 计算 =====================
def Linear_cnt_flops(module, input, output):
    input = input[0]
    pre_last = np.prod(input.shape[0:-1], dtype=np.int64)
    bias_flops = output.shape[-1] if module.bias is not None else 0
    module.__flops__ += int((input.shape[-1]*output.shape[-1] + bias_flops) * pre_last * (module.logger_w.ratio if hasattr(module,'logger_w') else 1))

layer_dict = {
    'gcn': GCNConvRaw,
    'gcn_rnd': GCNConvRnd,
    'gcn_thr': GCNConvThr,
    'gat': GATv2ConvRaw,
    'gat_rnd': GATv2ConvRnd,
    'gat_thr': GATv2ConvThr,
    'gin': GINConvRaw,
    'gcn2': GCNIIConvRaw,
    'gcn2_thr': GCNIIConvThr,
    'gsage': SAGEConvRaw,
    'gsage_thr': SAGEConvThr,
}

flops_modules_dict = {
    nn.Linear: Linear_cnt_flops,
    GCNConvRaw: GCNConvRaw.cnt_flops,
    GCNConvRnd: GCNConvRnd.cnt_flops,
    GCNConvThr: GCNConvThr.cnt_flops,
    GATv2ConvRaw: GATv2ConvRaw.cnt_flops,
    GATv2ConvRnd: GATv2ConvRnd.cnt_flops,
    GATv2ConvThr: GATv2ConvThr.cnt_flops,
    GCNIIConvRaw: GCNIIConvRaw.cnt_flops,
    GCNIIConvThr: GCNIIConvThr.cnt_flops,
    SAGEConvRaw: SAGEConvRaw.cnt_flops,
    SAGEConvThr: SAGEConvThr.cnt_flops,
}
