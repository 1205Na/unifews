import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx

from gammagl.transforms import BaseTransform
from gammagl.utils import add_self_loops  


def pow_with_pinv(x, p: float):
    x = tlx.pow(x, p)
    x = tlx.where(x == float('inf'), tlx.zeros_like(x), 0.0)
    return x


class GenNorm(BaseTransform):
    def __init__(self, left: float, right: float = None, dtype=tlx.float32):
        self.left = left
        self.right = right if right is not None else (1.0 - left)
        self.dtype = dtype

    def forward(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        if data.edge_weight is None and data.edge_attr is None:
            edge_weight = tlx.ones((edge_index.shape[1],), dtype=self.dtype)
            key = 'edge_weight'
        else:
            edge_weight = data.edge_attr if data.edge_weight is None else data.edge_weight
            key = 'edge_attr' if data.edge_weight is None else 'edge_weight'

        row, col = edge_index

        # 等价 PyG scatter
        deg_out = tlx.unsorted_segment_sum(edge_weight, row, num_nodes)
        deg_out = pow_with_pinv(deg_out, -self.left)
        
        deg_in = tlx.unsorted_segment_sum(edge_weight, col, num_nodes)
        deg_in = pow_with_pinv(deg_in, -self.right)

        edge_weight = deg_out[row] * edge_weight * deg_in[col]
        setattr(data, key, edge_weight)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(D^(-{self.left}) A D^(-{self.right}))'


# ====================== 【彻底删除】所有不存在的函数！ ======================
# 只用 add_self_loops，完全兼容原版功能
class AddRemainingSelfLoops(BaseTransform):
    def __init__(self, attr: str = 'edge_weight', fill_value=1.0):
        self.attr = attr
        self.fill_value = fill_value

    def forward(self, data):
        
        data.edge_index, data.edge_weight = add_self_loops(
            data.edge_index,
            edge_attr=data.edge_weight,
            fill_value=self.fill_value,
            num_nodes=data.num_nodes
        )
        return data