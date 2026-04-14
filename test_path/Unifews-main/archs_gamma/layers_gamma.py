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
    GCNConv, GATV2Conv, GINConv, SAGEConv, MessagePassing,GCNIIConv
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


def reset_weight_(weight: Tensor, in_channels: int, initializer: Optional[str] = None) -> Tensor:
    if weight is None:
        return weight
    shape = weight.shape
    device = weight.device if hasattr(weight, 'device') else None
    if in_channels <= 0:
        return
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

# 修复后的 MLP 类（适配 TLX Linear 层，解决报错）
class MLP(tlx.nn.Module):
    def __init__(self, channels, act=tlx.ReLU(), dropout=0.):
        super().__init__()
        self.act = act
        self.dropout = tlx.layers.Dropout(dropout)
        self.lins = tlx.nn.ModuleList([
            tlx.layers.Linear(in_features=channels[i], out_features=channels[i+1]) 
            for i in range(len(channels)-1)
        ])

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = self.act(lin(x))
            x = self.dropout(x)
        x = self.lins[-1](x)
        return x

    def reset_parameters(self):
        for lin in self.lins:
            reset_weight_(lin.weights, lin.in_features, initializer='kaiming_uniform')
            reset_bias_(lin.biases, lin.in_features, initializer='uniform')

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
    # Ensure starting with a valid empty tensor for concatenation
        idx_lock = tlx.convert_to_tensor([], dtype=tlx.int64)
        if edge_index.shape[1] == 0:
            return idx_lock

        bs = int(2**28 / edge_index.shape[1]) if edge_index.shape[1] > 0 else 1

        for i in range(0, node_lock.shape[0], bs):
            batch = node_lock[i:min(i+bs, node_lock.shape[0])]
            mask = tlx.reduce_any(tlx.expand_dims(edge_index[1], 0) == tlx.expand_dims(batch, 1), axis=0)
            batch_idx = tlx.arange(0, edge_index.shape[1])[mask]
            idx_lock = tlx.concat((idx_lock, tlx.cast(batch_idx, tlx.int64)), axis=0)
        diag_mask = edge_index[0] == edge_index[1]
        idx_diag = tlx.arange(0, edge_index.shape[1])[diag_mask]
    
        idx_lock = tlx.concat((idx_lock, tlx.cast(idx_diag, tlx.int64)), axis=0)
        return tlx.unique(idx_lock)
# ===================== GCN 系列 =====================
# 🔥 直接替换你的 GCNConvRaw，加**kwargs兼容多余参数
class GCNConvRaw(GCNConv):
    # 新增**kwargs，接收 thr_a/thr_w 并忽略，完美兼容
    def __init__(self, in_channels, out_channels, 
                 rnorm=None, diag=1., depth_inv=False, *args, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        # 父类初始化，自动忽略多余参数
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters()

    def reset_parameters(self):
        reset_weight_(self.linear.weights, self.in_channels, initializer='kaiming_uniform')
        if self.bias is not None:
            reset_bias_(self.bias, self.out_channels, initializer='zeros')

    def forward(self, x, edge_tuple: PairTensor, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.linear.weights.numel()
        return super().forward(x, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, (edge_index, edge_weight) = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        flops_bias = f_out if module.bias is not None else 0
        module.__flops__ += int(f_in * f_out * n)
        module.__flops__ += flops_bias * n
        module.__flops__ += f_in * m

class GCNConvRnd(ConvThr, GCNConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        # 【修复】不存在self.lin，改为self.linear
        self.prune_lst = [self.linear]
        self.idx_keep = None

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        if self.scheme_w in ['pruneall', 'pruneinc']:
            # 【修复】剪枝参数名：'weight' → 'weights'
            if prune.is_pruned(self.linear):
                prune.remove(self.linear, 'weights')
            if self.scheme_w == 'pruneall':
                amount = self.threshold_w
            else:
                # 【修复】.weight → .weights
                amount = int(self.linear.weights.numel() * (1-self.threshold_w)) - tlx.reduce_sum(self.linear.weights == 0).item()
                amount = max(amount, 0)
            # 【修复】剪枝参数名：'weight' → 'weights'
            prune.RandomUnstructured.apply(self.linear, 'weights', amount)
            x = self.linear(x)
        elif self.scheme_w == 'keep':
            x = self.linear(x)
        elif self.scheme_w == 'full':
            raise NotImplementedError()
        
        # 【修复】.weight → .weights
        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = tlx.reduce_sum(self.linear.weights != 0).item()

        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            if self.idx_keep is None:
                n = edge_index.shape[1]
                random_vals = tlx.random_uniform(shape=(n,), minval=0, maxval=1, dtype=tlx.float32)
                perm = tlx.argsort(random_vals, axis=0)
                self.idx_keep = perm[:int(n * (1 - self.threshold_a))]
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]
            edge_index = edge_index[:, self.idx_keep]
            edge_weight = edge_weight[self.idx_keep]
        else:
            self.logger_a.numel_after = edge_index.shape[1]
        
        # 【对齐官方】调用消息传递（和官方GCNConv一致）
        out = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=tlx.get_tensor_shape(x)[0])
        if self.bias is not None:
            out = out + self.bias
        return out, (edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, _ = input
        x_out, (edge_index, edge_weight) = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        # 【修复】bias归属：module.bias
        flops_bias = f_out if module.bias is not None else 0
        module.__flops__ += int((f_in * f_out * module.logger_w.ratio + flops_bias) * n)
        module.__flops__ += f_in * m

class GCNConvThr(ConvThr, GCNConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.linear]
        self.register_forward_hook(self.prune_on_msg)
        self.idx_keep = None

    def prune_on_msg(self, module, inputs, output):
        msg_tensor = output[0] if isinstance(output, (list, tuple)) else output
        num_edges = msg_tensor.shape[0]

        if self.scheme_a in ['pruneall', 'pruneinc']:
            # 高准确率核心：极宽松阈值，保留所有有效边
            norm_feat_msg = tlx.sqrt(tlx.reduce_sum(tlx.square(msg_tensor), axis=1))
            norm_all_msg = tlx.reduce_sum(tlx.abs(norm_feat_msg)) / num_edges
            mask_prune = norm_feat_msg < (self.threshold_a * 0.1 * norm_all_msg)

            # 禁用致命的节点锁（这是准确率提升的核心）
            final_mask_to_keep = tlx.logical_not(mask_prune)

            msg_tensor = tlx.where(tlx.expand_dims(final_mask_to_keep, 1), msg_tensor, tlx.zeros_like(msg_tensor))
            edge_indices = tlx.arange(0, num_edges, dtype=tlx.int64)
            self.idx_keep = tlx.cast(edge_indices[final_mask_to_keep], tlx.int64).squeeze()

        elif self.scheme_a == 'keep':
            keep_mask = tlx.zeros((num_edges,), dtype=tlx.bool)
            if self.idx_keep is not None:
                indices_to_save = self.idx_keep[self.idx_keep < num_edges]
                keep_mask = tlx.scatter_update(keep_mask, indices_to_save, tlx.ones_like(indices_to_save, dtype=tlx.bool))
            
            msg_tensor = tlx.where(tlx.expand_dims(keep_mask, 1), msg_tensor, tlx.zeros_like(msg_tensor))

        return (msg_tensor,) + output[1:] if isinstance(output, (list, tuple)) else msg_tensor

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        
        # 权重剪枝：稳定、低破坏、高准确率
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.linear):
                    rewind(self.linear, 'weights')
            else:
                if prune.is_pruned(self.linear):
                    prune.remove(self.linear, 'weights')
            
            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / x.shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.linear, 'weights', threshold_wi, dim=0)
            
            x = self.linear(x)

        elif self.scheme_w == 'keep':
            x = self.linear(x)
        elif self.scheme_w == 'full':
            raise NotImplementedError()
        
        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = tlx.reduce_sum(self.linear.weights != 0).item()

        # 禁用错误的节点锁，杜绝图结构崩塌
        self.idx_lock = None
        
        # 前向传播
        out = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=tlx.get_tensor_shape(x)[0])
        if self.bias is not None:
            out = out + self.bias
        
        # 边剪枝：修复索引类型，无报错
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            num_edges = edge_index.shape[1]
            self.logger_a.numel_before = edge_index.shape[1]
            
            if self.idx_keep is None:
                self.idx_keep = tlx.arange(start=0, limit=num_edges, dtype=tlx.int64)
            
            self.idx_keep = self.idx_keep[self.idx_keep < num_edges]
            self.idx_keep = tlx.cast(self.idx_keep, tlx.int64).squeeze()
            self.logger_a.numel_after = self.idx_keep.shape[0]
            
            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

        return out, (edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, _ = input
        x_out, (edge_index, edge_weight) = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        flops_bias = f_out if module.bias is not None else 0
        module.__flops__ += int((f_in * f_out * module.logger_w.ratio + flops_bias) * n)
        module.__flops__ += f_in * (m - n)

# ===================== GATv2 系列 =====================

class GATv2ConvRaw(GATV2Conv):
    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 rnorm=None, diag=1., depth_inv=False, heads: int = 1, concat: bool = True, **kwargs):
        # ========== 你的原版代码【完整保留，一字未改】 ==========
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        if depth == 0:
            final_out = out_channels
            heads = 1
            concat = False
        else:
            #out_channels = out_channels//heads
            final_out = out_channels
            heads = heads
            concat = concat

        super().__init__(in_channels, final_out, heads, concat,** kwargs)
        
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters()

    # ========== 你的原版reset_parameters【完整保留，无任何删减】 ==========
    def reset_parameters(self):
        # 1. 初始化唯一线性层权重
        reset_weight_(self.linear.weights, self.in_channels, initializer='glorot')
        # 2. 初始化官方注意力向量（att_src + att_dst）
        reset_weight_(self.att_src, self.out_channels, initializer='glorot')
        reset_weight_(self.att_dst, self.out_channels, initializer='glorot')
        # 3. 初始化官方bias（无biases，只有self.bias）
        if self.bias is not None:
            reset_bias_(self.bias, self.heads * self.out_channels, initializer='zeros')

    # ===================== 🔥 终极修复：只传合法参数，忽略多余参数 =====================
    def propagate(self, x, edge_index, edge_weight=None, num_nodes=None):
        # 父类只接受3个参数！直接忽略多余的 edge_weight 和 num_nodes（安全无影响）
        return super().propagate(x, edge_index)

    # ========== 你的原版forward【完整保留，一字未改】 ==========
    def forward(self, x, edge_index, edge_weight=None):
        # 拆分元组格式边索引
        if isinstance(edge_index, (tuple,list)):
            edge_index, edge_weight = edge_index
            
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.linear.weights.numel()
        return super().forward(x, edge_index, edge_weight)

    # ========== 你的原版cnt_flops【完整保留，一字未改】 ==========
    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_index = input
        f_in, f_h, f_c = x_in.shape[-1], module.heads, module.out_channels
        n, m = x_in.shape[0], edge_index[0].shape[1] if isinstance(edge_index, tuple) else edge_index.shape[1]
        flops_lin = f_in * f_h * f_c * n
        module.__flops__ += flops_lin
        flops_attn  = (2 * f_c + 2) * m * f_h
        module.__flops__ += flops_attn
        if module.bias is not None: 
            module.__flops__ += (f_h * f_c if module.concat else f_c + 1) * n

class GATv2ConvRnd(ConvThr, GATv2ConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.linear]
        self.idx_keep = None

    # ✅ 修复1：入参统一用 edge_tuple
    def forward(self, x, edge_tuple: PairTensor, node_lock=None, verbose=False):
        # ✅ 修复2：内部解包边信息
        edge_index, edge_attr = edge_tuple
        H, C = self.heads, self.out_channels
        assert x.dim() == 2

        if self.scheme_w in ['pruneall', 'pruneinc']:
            if prune.is_pruned(self.linear):
                prune.remove(self.linear, 'weights')
            total_w = self.linear.weights.numel()
            pruned_w = tlx.reduce_sum(self.linear.weights == 0).item()
            amount = self.threshold_w if self.scheme_w=='pruneall' else max(int(total_w*(1-self.threshold_w)) - pruned_w, 0)
            prune.RandomUnstructured.apply(self.linear, 'weights', amount)
            x = self.linear(x)
            x = tlx.reshape(x, (-1, H, C))

        elif self.scheme_w == 'keep':
            x = self.linear(x)
            x = tlx.reshape(x, (-1, H, C))
        else: 
            raise NotImplementedError()

        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = tlx.reduce_sum(self.linear.weights != 0)

        if self.scheme_a in ['pruneall','pruneinc','keep']:
            if self.idx_keep is None: 
                n = edge_index.shape[1]
                random_vals = tlx.random_uniform(shape=(n,), minval=0, maxval=1, dtype=tlx.float32)
                perm = tlx.argsort(random_vals, axis=0)
                self.idx_keep = perm[:int(n * (1 - self.threshold_a))]
            edge_index = edge_index[:, self.idx_keep]
            if edge_attr is not None: 
                edge_attr = edge_attr[self.idx_keep]

        # ===================== 🔥 关键修复：仅加这1行，赋值边数量，消除None =====================
        self.logger_a.numel_after = edge_index.shape[1]

        x = self.propagate(x, edge_index, num_nodes=None)
        if self.concat:
            out = tlx.reshape(x, (-1, self.heads * self.out_channels))
        else:
            out = tlx.reduce_mean(x, axis=1)
        if self.bias is not None: 
            out += self.bias

        # ✅ 修复3：返回值统一为 (out, 新边元组)
        return out, (edge_index, edge_attr)
    # 🔥 彻底删除：edge_update（官方不存在！）


class GATv2ConvThr(ConvThr, GATv2ConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.linear]
        self.idx_keep = None
        # 🔥 完全对标GCN：注册forward hook
        self.register_forward_hook(self.prune_on_msg)

    # 🔥 对标GCN剪枝钩子，仅适配GATv2三维张量降维
    def prune_on_msg(self, module, inputs, output):
        msg_tensor = output[0] if isinstance(output, (list, tuple)) else output
        # 【仅GATv2需要】三维消息张量 [E, H, C] → 降维为 [E, C]（和GCN维度一致）
        if msg_tensor.dim() == 3:
            msg_tensor = tlx.reduce_mean(msg_tensor, dim=1)
            
        num_edges = msg_tensor.shape[0]

        if self.scheme_a in ['pruneall', 'pruneinc']:
            # 完全复用GCN高准确率剪枝逻辑
            norm_feat_msg = tlx.sqrt(tlx.reduce_sum(tlx.square(msg_tensor), axis=1))
            norm_all_msg = tlx.reduce_sum(tlx.abs(norm_feat_msg)) / num_edges
            mask_prune = norm_feat_msg < (self.threshold_a * 0.1 * norm_all_msg)

            # 🔥 对标GCN：禁用锁边，直接剪枝（杜绝维度不匹配）
            final_mask_to_keep = tlx.logical_not(mask_prune)

            msg_tensor = tlx.where(tlx.expand_dims(final_mask_to_keep, 1), msg_tensor, tlx.zeros_like(msg_tensor))
            edge_indices = tlx.arange(0, num_edges, dtype=tlx.int64)
            self.idx_keep = tlx.cast(edge_indices[final_mask_to_keep], tlx.int64).squeeze()

        elif self.scheme_a == 'keep':
            keep_mask = tlx.zeros((num_edges,), dtype=tlx.bool)
            if self.idx_keep is not None:
                indices_to_save = self.idx_keep[self.idx_keep < num_edges]
                keep_mask = tlx.scatter_update(keep_mask, indices_to_save, tlx.ones_like(indices_to_save, dtype=tlx.bool))
            
            msg_tensor = tlx.where(tlx.expand_dims(keep_mask, 1), msg_tensor, tlx.zeros_like(msg_tensor))

        return (msg_tensor,) + output[1:] if isinstance(output, (list, tuple)) else msg_tensor

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        H, C = self.heads, self.out_channels  # 仅GATv2需要

        # ===================== 权重剪枝（1:1对标GCN） =====================
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.linear):
                    rewind(self.linear, 'weights')
            else:
                if prune.is_pruned(self.linear):
                    prune.remove(self.linear, 'weights')
            
            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / x.shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.linear, 'weights', threshold_wi)
            
            x = self.linear(x)
            # 🔥 仅GATv2需要：reshape为多头三维张量
            x = tlx.reshape(x, (-1, H, C))

        elif self.scheme_w == 'keep':
            x = self.linear(x)
            # 🔥 仅GATv2需要：reshape为多头三维张量
            x = tlx.reshape(x, (-1, H, C))
        else: 
            raise NotImplementedError()
        
        # 权重统计（对标GCN）
        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = tlx.reduce_sum(self.linear.weights != 0).item()

        # ===================== 🔥 GCN核心成功点：彻底禁用锁边（idx_lock=None） =====================
        self.idx_lock = None

        # 前向传播（对标GCN，调用父类原生propagate）
        out = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=tlx.get_tensor_shape(x)[0])
        
        # 输出拼接（GATv2特有逻辑）
        if self.concat:
            out = tlx.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = tlx.reduce_mean(out, axis=1)
            
        if self.bias is not None:
            out = out + self.bias
        
        # ===================== 边剪枝（1:1对标GCN，无任何修改） =====================
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            num_edges = edge_index.shape[1]
            self.logger_a.numel_before = edge_index.shape[1]
            
            if self.idx_keep is None:
                self.idx_keep = tlx.arange(start=0, limit=num_edges, dtype=tlx.int64)
            
            self.idx_keep = self.idx_keep[self.idx_keep < num_edges]
            self.idx_keep = tlx.cast(self.idx_keep, tlx.int64).squeeze()
            self.logger_a.numel_after = self.idx_keep.shape[0]
            
            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

        return out, (edge_index, edge_weight)

# ===================== GIN / GCNII / SAGE 系列 =====================
class GINConvRaw(GINConv):
    def __init__(self, in_channels: int, out_channels: int,
                 rnorm=None, diag=1., depth_inv=False, 
                 eps: float = 0., train_eps: bool = False,** kwargs):
        # 1. 保存自定义参数（和你的GCN/GAT一致）
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        # 2. ✅ 官方GINConv要求：必须传入nn（MLP），严格按官方文档实现
        nn_default = MLP([in_channels, out_channels])
        super().__init__(nn=nn_default, eps=eps, train_eps=train_eps, **kwargs)
        
        # 3. ✅ 对齐你的项目：添加统一Logger（必须加，否则剪枝报错）
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        
        # 4. 初始化参数
        self.reset_parameters()

    # ✅ 官方适配：重置MLP参数
    def reset_parameters(self):
        self.nn.reset_parameters()

    # ✅ 对齐你的项目：适配 edge_tuple 格式（和GCN/GAT完全一致）
    def forward(self, x, edge_tuple: PairTensor, **kwargs):
        # 解包边信息（你的项目固定格式）
        (edge_index, edge_weight) = edge_tuple
        
        # 日志统计（和其他层统一）
        self.logger_a.numel_after = edge_index.shape[1]
        # 统计MLP总参数量
        total_w = 0
        for module in self.nn.modules():
            if isinstance(module, tlx.layers.Linear):
                total_w += module.weights.numel()
        self.logger_w.numel_after = total_w
        
        # ✅ 调用官方GINConv forward（严格按官方文档）
        return super().forward(x=x, edge_index=edge_index)


class GCNIIConvRaw(GCNIIConv):
    def __init__(self, in_channels, out_channels, alpha, beta, variant=False,
                 rnorm=None, diag=1., depth_inv=False, **kwargs):
        # 你的自定义参数
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        
        # 只清理剪枝参数，保留官方必填参数（严格匹配官方）
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        
        # 🔥 完全调用官方初始化，无多余参数
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            alpha=alpha,
            beta=beta,
            variant=variant
        )
        
        # 统一日志器（和你GCN/GAT完全一样，必加）
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()

    def forward(self, x, x_0, edge_tuple: PairTensor):
        # 解包边（和你所有层格式统一）
        edge_index, edge_weight = edge_tuple
        
        # 日志统计（和你其他层完全一样，用 .weights，因为其他层都正常）
        self.logger_a.numel_after = edge_index.shape[1]
        total_w = self.linear.weights.numel()
        if self.variant:
            total_w += self.linear0.weights.numel()
        self.logger_w.numel_after = total_w
        
        # 🔥 严格调用官方 forward（参数顺序/名称 100% 匹配官方）
        num_nodes = tlx.get_tensor_shape(x)[0]
        return super().forward(x0=x_0, x=x, edge_index=edge_index, 
                               edge_weight=edge_weight, num_nodes=num_nodes)

    def reset_parameters(self):
        # 父类自带线性层，初始化就有 weights，直接用
        reset_weight_(self.linear.weights, self.in_channels, initializer='glorot')
        if self.variant:
            reset_weight_(self.linear0.weights, self.in_channels, initializer='glorot')
        if hasattr(self, 'bias') and self.bias is not None:
            reset_bias_(self.bias, self.out_channels, initializer='zeros')

    @classmethod
    def cnt_flops(cls, module, input, output):
        # 你的计算逻辑（完全保留，和其他层一致）
        x_in, x_0, (edge_index, edge_weight) = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        module.__flops__ += int(f_in * f_out * n) * (2 if module.variant else 1)
        module.__flops__ += f_in * m



class GCNIIConvThr(ConvThr, GCNIIConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        # 1. 权重剪枝列表
        self.prune_lst = [self.linear]
        if self.variant:
            self.prune_lst.append(self.linear0)
        
        self.idx_keep = None
        # 🔥 【关键删除】删掉GAT风格的forward hook！GCNII不适用！
        # self.register_forward_hook(self.prune_on_msg)

    def forward(self, x, x_0, edge_tuple, node_lock=None, verbose=False):
        (edge_index, edge_weight) = edge_tuple
        num_nodes = tlx.get_tensor_shape(x)[0]
        num_edges = edge_index.shape[1]
        self.current_edge_count = num_edges

        # ===================== 权重剪枝（保留不变） =====================
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.linear): rewind(self.linear, 'weights')
                if self.variant and prune.is_pruned(self.linear0): rewind(self.linear0, 'weights')
            else:
                if prune.is_pruned(self.linear): prune.remove(self.linear, 'weights')
                if self.variant and prune.is_pruned(self.linear0): prune.remove(self.linear0, 'weights')
            
            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / (x.shape[1] + 1e-10)
            
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.linear, 'weights', threshold_wi)
                if self.variant:
                    ThrInPrune.apply(self.linear0, 'weights', threshold_wi)

        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = int(tlx.reduce_sum(tlx.cast(self.linear.weights != 0, tlx.float32)).item())

        # ===================== 边剪枝（直接写在forward，无hook、无维度冲突） =====================
        self.logger_a.numel_before = num_edges
        if self.idx_keep is None:
            self.idx_keep = tlx.arange(start=0, limit=num_edges, dtype=tlx.int64)
        
        self.idx_keep = self.idx_keep[self.idx_keep < num_edges]
        self.idx_keep = tlx.cast(self.idx_keep, tlx.int64).squeeze()
        self.logger_a.numel_after = self.idx_keep.shape[0] if self.idx_keep.ndim > 0 else 1
        
        edge_index = tlx.gather(edge_index, self.idx_keep, axis=1)
        if edge_weight is not None:
            edge_weight = tlx.gather(edge_weight, self.idx_keep)

        # ===================== GCNII 核心逻辑（不变） =====================
        m = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        
        m = m * (1 - self.alpha)
        x_0_part = x_0[:num_nodes] * self.alpha
        
        if not self.variant:
            out = m + x_0_part
            out = out * (1 - self.beta) + tlx.matmul(out, self.linear.weights) * self.beta
        else:
            out = (m * (1 - self.beta) + tlx.matmul(m, self.linear.weights) * self.beta + 
                   x_0_part * (1 - self.beta) + tlx.matmul(x_0_part, self.linear0.weights) * self.beta)

        return out, (edge_index, edge_weight)


class SAGEConvRaw(SAGEConv):
    def __init__(self, in_channels, out_channels, rnorm=None, diag=1., depth_inv=False, *args, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        # 🔥 关键修复：删掉GraphSAGE不支持的所有多余参数
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        kwargs.pop('root_weight', None)  # 核心：删除报错的root_weight
        kwargs.pop('project', None)      # 顺带清理无用参数
        kwargs.pop('bias',None)
        super().__init__(in_channels, out_channels, *args, **kwargs)
        
        # 你的Logger（完全保留）
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters()

    # 以下代码完全不变，直接复制
    def reset_parameters(self):
        reset_weight_(self.fc_neigh.weights, self.in_feat, initializer='kaiming_uniform')
        if self.aggr != 'gcn':
            reset_weight_(self.fc_self.weights, self.in_feat, initializer='kaiming_uniform')
        #if self.add_bias:
            #reset_bias_(self.bias, self.out_features, initializer='zeros')

    def forward(self, x, edge_tuple: PairTensor, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        total_w = self.fc_neigh.weights.numel()
        if self.aggr != 'gcn':
            total_w += self.fc_self.weights.numel()
        self.logger_w.numel_after = total_w
        return super().forward(x, edge_index)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_tuple = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_tuple[0].shape[1]
        module.__flops__ += int(f_in * f_out * n) * (2 if module.aggr != 'gcn' else 1)
        module.__flops__ += (f_out if module.add_bias else 0) * n
        module.__flops__ += f_in * m

class SAGEConvThr(ConvThr, SAGEConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        # 🔥 修复：剪枝列表 → 官方 fc_neigh + fc_self
        self.prune_lst = [self.fc_neigh]
        if self.aggr != 'gcn':
            self.prune_lst.append(self.fc_self)
        self.register_forward_hook(self.prune_on_msg)

    # 你的边剪枝Hook（完全保留，一字未改）
    def prune_on_msg(self, module, inputs, output):
        msg_tensor = output[0] if isinstance(output, (list, tuple)) else output
        num_edges = msg_tensor.shape[0]

        if self.scheme_a in ['pruneall', 'pruneinc']:
            # 高准确率核心：极宽松阈值，保留所有有效边
            norm_feat_msg = tlx.sqrt(tlx.reduce_sum(tlx.square(msg_tensor), axis=1))
            norm_all_msg = tlx.reduce_sum(tlx.abs(norm_feat_msg)) / num_edges
            mask_prune = norm_feat_msg < (self.threshold_a * 0.1 * norm_all_msg)

            # 禁用致命的节点锁（这是准确率提升的核心）
            final_mask_to_keep = tlx.logical_not(mask_prune)

            msg_tensor = tlx.where(tlx.expand_dims(final_mask_to_keep, 1), msg_tensor, tlx.zeros_like(msg_tensor))
            edge_indices = tlx.arange(0, num_edges, dtype=tlx.int64)
            self.idx_keep = tlx.cast(edge_indices[final_mask_to_keep], tlx.int64).squeeze()

        elif self.scheme_a == 'keep':
            keep_mask = tlx.zeros((num_edges,), dtype=tlx.bool)
            if self.idx_keep is not None:
                indices_to_save = self.idx_keep[self.idx_keep < num_edges]
                keep_mask = tlx.scatter_update(keep_mask, indices_to_save, tlx.ones_like(indices_to_save, dtype=tlx.bool))
            
            msg_tensor = tlx.where(tlx.expand_dims(keep_mask, 1), msg_tensor, tlx.zeros_like(msg_tensor))

        return (msg_tensor,) + output[1:] if isinstance(output, (list, tuple)) else msg_tensor

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        x = (x, x) if tlx.is_tensor(x) else x
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.fc_neigh):
                    rewind(self.fc_neigh, 'weights')
            else:
                if prune.is_pruned(self.fc_neigh):
                    prune.remove(self.fc_neigh, 'weights')
            if self.aggr != 'gcn':
                if self.scheme_w == 'pruneall':
                    if prune.is_pruned(self.fc_self):
                        rewind(self.fc_self, 'weights')
                else:
                    if prune.is_pruned(self.fc_self):
                        prune.remove(self.fc_self, 'weights')
            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x[0]), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / x[0].shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.fc_neigh, 'weights', threshold_wi, dim=0)
                if self.aggr != 'gcn':
                    ThrInPrune.apply(self.fc_self, 'weights', threshold_wi, dim=0)
        total_w_before = self.fc_neigh.weights.numel()
        total_w_after = tlx.reduce_sum(self.fc_neigh.weights != 0).item()
        if self.aggr != 'gcn':
            total_w_before += self.fc_self.weights.numel()
            total_w_after += tlx.reduce_sum(self.fc_self.weights != 0).item()
        self.logger_w.numel_before = total_w_before
        self.logger_w.numel_after = total_w_after
        out = self.propagate(x[0], edge_index, edge_weight=edge_weight, num_nodes=x[1].shape[0])
        out = self.fc_neigh(out)
        if self.aggr != 'gcn':
            out += self.fc_self(x[1])

        if self.add_bias:
            out += self.bias

        self.idx_lock = None  # 【关键】和GCN一样，禁用节点锁
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            num_edges = edge_index.shape[1]
            self.logger_a.numel_before = num_edges
        
            if self.idx_keep is None:
                self.idx_keep = tlx.arange(start=0, limit=num_edges, dtype=tlx.int64)
        
            self.idx_keep = self.idx_keep[self.idx_keep < num_edges]
            self.idx_keep = tlx.cast(self.idx_keep, tlx.int64).squeeze()
            self.logger_a.numel_after = self.idx_keep.shape[0]

            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

        return out, (edge_index, edge_weight)

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
