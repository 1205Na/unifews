import os
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
import tensorlayerx.nn as nn
# IMPORT YOUR CUSTOM FUNCTIONS HERE:
from .layers_gamma import (
    layer_dict, ThrInPrune, LayerNumLogger, rewind, 
    reset_weight_, reset_bias_, 
    gcn_norm, add_remaining_self_loops  # <-- Added your normalization functions
)

kwargs_default = {
    'gcn': {
        'cached': False,
        'add_self_loops': False,
        'improved': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.0,
        'depth_inv': False,
    },
    'gin': {
        'eps': 0.0,
        'train_eps': False,
        'rnorm': None,
        'diag': 1.0,
        'depth_inv': False,
    },
    'gat': {
        'heads': 8,
        'concat': True,
        'share_weights': False,
        'add_self_loops': False,
        'rnorm': None,
        'diag': 1.0,
        'depth_inv': False,
    },
    'gcn2': {
        'alpha': 0.1,
        'theta': 0.5,
        'shared_weights': True,
        'cached': False,
        'add_self_loops': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.0,
        'depth_inv': True,
    },
    'gsage': {
        'aggr': 'mean',
        'improved': False,
        'normalize': False,
        'root_weight': True,
        'project': False,
        'bias': True,
        'rnorm': 0.5,
        'diag': 1.0,
        'depth_inv': False,
    },
}

def state2module(model, name):
    parts = name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module

def set_attr(module, key, value):
    if hasattr(module, key):
        setattr(module, key, value)

class GNNThr(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass, thr_a=0.0, thr_w=0.0, dropout=0.0, layer='gcn', **kwargs):
        super().__init__()
        self.apply_thr = '_' in layer
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.use_bn = True
        self.kwargs = kwargs

        Conv = layer_dict[layer]
        for k, v in kwargs_default[layer.split('_')[0]].items():
            self.kwargs.setdefault(k, v)

        if not isinstance(thr_a, list):
            if layer.endswith('_rnd'):
                thr_a = [thr_a] + [0.0]*(nlayer-1)
            else:
                thr_a = [thr_a]*nlayer
        thr_w = thr_w if isinstance(thr_w, list) else [thr_w]*nlayer

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # ====================================================================
        # [ACCURACY FIX] Extract graph operation flags into class attributes.
        # This prevents GammaGL's __init__ from crashing while allowing us 
        # to process the graph mathematically exactly like PyG would.
        # ====================================================================
        self.depth_inv = self.kwargs.pop('depth_inv', False)
        self.normalize_adj = self.kwargs.pop('normalize', False)
        self.add_self_loops = self.kwargs.pop('add_self_loops', False)
        self.cached = self.kwargs.pop('cached', False)
        
        conv_kwargs = self.kwargs.copy()
        # Remove remainder args that standard GammaGL Convs don't support
        for k in ['improved', 'rnorm', 'diag']:
            conv_kwargs.pop(k, None)

        self.convs.append(Conv(nfeat, nhidden, thr_a=thr_a[0], thr_w=thr_w[0], **conv_kwargs))
        self.norms.append(nn.BatchNorm1d(num_features=nhidden, momentum=0.1))
        
        for i in range(1, nlayer-1):
            self.convs.append(Conv(nhidden, nhidden, thr_a=thr_a[i], thr_w=thr_w[i], **conv_kwargs))
            self.norms.append(nn.BatchNorm1d(num_features=nhidden, momentum=0.1))
        
        self.convs.append(Conv(nhidden, nclass, thr_a=thr_a[-1], thr_w=thr_w[-1], **conv_kwargs))

    def reset_parameters(self):
        # 1. Reset Convolutions
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        
        # 2. Reset Normalization Layers Safely
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                # If the norm layer (like LayerNorm) HAS the method, use it
                norm.reset_parameters()
            elif 'BatchNorm' in type(norm).__name__:
                # If it's a TLX BatchNorm, use our manual reset helper
                from .layers_gamma import reset_bn_
                reset_bn_(norm)

    # ====================================================================
    # [NEW GRAPH PROCESSOR] Dynamically applies your layer_gamma.py functions
    # ====================================================================
    def _process_graph(self, x, edge_idx):
        if not (self.normalize_adj or self.add_self_loops):
            return edge_idx
            
        # Unpack tuple if it came from identity_n_norm
        if isinstance(edge_idx, (tuple, list)):
            edge_index, edge_weight = edge_idx[0], edge_idx[1]
        else:
            edge_index, edge_weight = edge_idx, None

        num_nodes = x.shape[0]

        # Apply your exact functional logic
        if self.normalize_adj:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, num_nodes, 
                add_self_loops=self.add_self_loops, dtype=x.dtype
            )
        elif self.add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
            )
            
        return (edge_index, edge_weight)

    def forward(self, x, edge_idx, node_lock=tlx.convert_to_tensor([]), verbose=False):
        # 1. Manually route edges through your custom PyG-equivalent math
        edge_idx = self._process_graph(x, edge_idx)
        
        # 2. Proceed with normal forward pass
        if self.apply_thr:
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, edge_idx, node_lock=node_lock, verbose=verbose)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[-1](x, edge_idx)
        return x

    def get_repre(self, x, edge_idx, layer=None, node_lock=tlx.convert_to_tensor([]), verbose=False):
        layer = layer or len(self.convs)-1
        edge_idx = self._process_graph(x, edge_idx)
        
        if self.apply_thr:
            for i, conv in enumerate(self.convs[:layer]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[layer](x, edge_idx, node_lock=node_lock, verbose=verbose)
        else:
            for i, conv in enumerate(self.convs[:layer]):
                x = conv(x, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[layer](x, edge_idx)
        return x

    def set_scheme(self, scheme_a, scheme_w):
        self.apply(lambda m: set_attr(m, 'scheme_a', scheme_a))
        self.apply(lambda m: set_attr(m, 'scheme_w', scheme_w))

    def remove(self):
        for conv in self.convs:
            if hasattr(conv, 'prune_lst'):
                for m in conv.prune_lst:
                    if tlx.nn.utils.prune.is_pruned(m):
                        tlx.nn.utils.prune.remove(m, 'weight')

    def get_numel(self):
        numel_a = sum(c.logger_a.numel_after for c in self.convs)
        numel_w = sum(c.logger_w.numel_after for c in self.convs)
        return numel_a/1e3, numel_w/1e3

    @classmethod
    def batch_counter_hook(cls, module, inp, out):
        if not hasattr(module, '__batch_counter__'):
            module.__batch_counter__ = 0
        module.__batch_counter__ += 1

class GNNLPThr(GNNThr):
    def __init__(self, nlayer, nfeat, nhidden, nclass, thr_a=0.0, thr_w=0.0, dropout=0.0, layer='gcn', **kwargs):
        super().__init__(nlayer, nfeat, nhidden, nhidden, thr_a, thr_w, dropout, layer,** kwargs)
        self.lin_out = nn.ModuleList([
            nn.Linear(nhidden, nhidden),
            nn.Linear(nhidden, nhidden),
            nn.Linear(nhidden, 1),
        ])

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lin_out:
            reset_weight_(lin.weights, lin.in_features, initializer='kaiming_uniform')
            reset_bias_(lin.biases, lin.in_features, initializer='uniform')

    def decode(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lin_out[:-1]:
            x = lin(x)
            x = self.act(x)
            x = self.dropout(x)
        return self.lin_out[-1](x)

class SandwitchThr(GNNThr):
    def __init__(self, nlayer, nfeat, nhidden, nclass, thr_a=0.0, thr_w=0.0, dropout=0.0, layer='gcn', **kwargs):
        super().__init__(nlayer, nhidden, nhidden, nhidden, thr_a, thr_w, dropout, layer, **kwargs)
        self.lin_in = nn.Linear(nfeat, nhidden)
        self.lin_out = nn.Linear(nhidden, nclass)
        self.norms.append(nn.BatchNorm1d(num_features=nhidden, momentum=0.1))

    def reset_parameters(self):
        super().reset_parameters()
        reset_weight_(self.lin_in.weights, self.lin_in.in_features, initializer='kaiming_uniform')
        reset_bias_(self.lin_in.biases, self.lin_in.in_features, initializer='uniform')
        reset_weight_(self.lin_out.weights, self.lin_out.in_features, initializer='kaiming_uniform')
        reset_bias_(self.lin_out.biases, self.lin_out.in_features, initializer='uniform')

    def forward(self, x, edge_idx, node_lock=tlx.convert_to_tensor([]), verbose=False):
        # 1. Process edges through your PyG math equivalence
        edge_idx = self._process_graph(x, edge_idx)
        
        x = self.lin_in(x)
        if self.use_bn:
            x = self.norms[-1](x)
        x = self.act(x)
        x = self.dropout(x)

        if self.apply_thr:
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, x, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, x, edge_idx, node_lock=node_lock, verbose=verbose)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, x, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[-1](x, x, edge_idx)
        return self.lin_out(x)

class MLP(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass, dropout, thr_w=0.0, layer='sgc'):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.threshold_w = thr_w
        self.scheme_w = 'full'

        self.fcs = nn.ModuleList()
        if nlayer == 1:
            self.fcs.append(nn.Linear(nfeat, nclass))
        else:
            self.fcs.append(nn.Linear(nfeat, nhidden))
            for _ in range(nlayer-2):
                self.fcs.append(nn.Linear(nhidden, nhidden))
            self.fcs.append(nn.Linear(nhidden, nclass))
        
        for fc in self.fcs:
            fc.logger_w = LayerNumLogger(layer)
            object.__setattr__(fc, 'act', lambda x: x)
            fc.bias = fc._bias if hasattr(fc, '_bias') else None

    def reset_parameters(self):
        for lin in self.fcs:
            reset_weight_(lin.weights, lin.in_features, initializer='kaiming_uniform')
            reset_bias_(lin.biases, lin.in_features, initializer='uniform')
            
    def apply_prune(self, lin, x):
        log = lin.logger_w
        log.numel_before = 1
        log.numel_after = 1

    def forward(self, x, edge_idx=None, *args, **kwargs):
        for i, fc in enumerate(self.fcs[:-1]):
            self.apply_prune(fc, x)
            x = fc(x)
            x = self.act(x)
            x = self.dropout(x)
        self.apply_prune(self.fcs[-1], x)
        return self.fcs[-1](x)

    def set_scheme(self, scheme_a, scheme_w):
        self.scheme_w = scheme_w

    def get_numel(self):
        return 0, sum(fc.logger_w.numel_after for fc in self.fcs)/1e3