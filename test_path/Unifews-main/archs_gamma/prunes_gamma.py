# 纯 TensorLayerX 实现，无 PyTorch 剪枝包依赖
import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module


# ===================== 【新增】剪枝基类 & 模拟 PyTorch prune 工具 =====================
class BasePruningMethod:
    """
    等价于 PyTorch 的 BasePruningMethod
    纯 TLX 实现，无框架依赖
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask=None):
        raise NotImplementedError("子类必须实现 compute_mask")

    @classmethod
    def apply(cls, module, name, *args, **kwargs):
        return cls(*args, **kwargs)


class prune:
    """
    【核心】自定义 prune 工具类
    完全对齐 PyTorch API：
        prune.is_pruned(module)
        prune.remove(module, name)
    """

    @staticmethod
    def is_pruned(module: Module) -> bool:
        """判断模块是否被剪枝"""
        return any(key.endswith("_mask") for key in dir(module))

    @staticmethod
    def remove(module: Module, name: str):
        """移除剪枝，恢复权重（等价 PyTorch prune.remove）"""
        rewind(module, name)


# ===================== 原有剪枝工具函数（完全不变） =====================
def prune_threshold(x, threshold=1e-3):
    """
    阈值剪枝：接口/逻辑与原版完全一致
    """
    norm_vals = tlx.norm(x, axis=1) / x.shape[1]
    idx_0 = norm_vals < threshold
    x = tlx.where(idx_0, tlx.zeros_like(x), x)
    return x, idx_0


def prune_topk(x, k=0.2):
    """
    TopK 剪枝：接口/逻辑与原版完全一致
    """
    num_0 = int(x.shape[0] * k)
    x_norm = tlx.norm(x, axis=1)
    _, idx_0 = tlx.topk(x_norm, num_0)
    mask = tlx.ones(x.shape[0], dtype=tlx.bool)
    mask[idx_0] = False
    x = tlx.where(mask[:, None], x, tlx.zeros_like(x))
    return x, idx_0


def rewind(module: Module, name: str):
    """
    恢复剪枝参数：兼容 TLX Module，逻辑与原版一致
    移除剪枝掩码，恢复原始参数
    """
    orig_name = name + "_orig"
    if hasattr(module, orig_name):
        setattr(module, name, getattr(module, orig_name))
        delattr(module, orig_name)
        if hasattr(module, name + "_mask"):
            delattr(module, name + "_mask")

    if hasattr(module, '_forward_pre_hooks'):
        hooks = module._forward_pre_hooks
        for k in list(hooks.keys()):
            hook = hooks[k]
            if hasattr(hook, '_tensor_name') and hook._tensor_name == name:
                del hooks[k]


# ===================== 剪枝类：继承 BasePruningMethod =====================
class ThrInPrune(BasePruningMethod):
    """输入维度阈值剪枝：接口/功能 100% 对齐原版"""
    PRUNING_TYPE = 'structured'

    def __init__(self, threshold, dim=0):
        self.threshold = threshold
        self.dim = dim

    def compute_mask(self, t):
        """计算剪枝掩码"""
        assert self.threshold.shape == t.shape[1:]
        tmax = tlx.reduce_max(tlx.abs(t)) * (1 - 1e-3)
        self.threshold = tlx.where(self.threshold > tmax, tmax, self.threshold)
        mask = tlx.ones_like(t)
        mask = tlx.where(tlx.abs(t) < self.threshold, 0.0, mask)
        return mask

    @classmethod
    def apply(cls, module: Module, name: str, threshold):
        pruner = cls(threshold)
        weight = getattr(module, name)
        mask = pruner.compute_mask(weight)

        setattr(module, name + "_orig", weight)
        setattr(module, name + "_mask", mask)
        setattr(module, name, weight * mask)
        return pruner


class ThrProdPrune(BasePruningMethod):
    """权重-输入乘积剪枝：接口/功能 100% 对齐原版"""
    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, t, importance_scores=None):
        if importance_scores is not None:
            t = importance_scores
        mask = tlx.ones_like(t)
        mask = tlx.where(tlx.abs(t) < self.threshold, 0.0, mask)
        return mask

    @classmethod
    def apply(cls, module: Module, name: str, threshold, x):
        w = getattr(module, name)
        assert w.shape[1] == x.shape[1]
        score = tlx.abs(w) * tlx.norm(x, axis=0)
        pruner = cls(threshold)
        mask = pruner.compute_mask(w, importance_scores=score)

        setattr(module, name + "_orig", w)
        setattr(module, name + "_mask", mask)
        setattr(module, name, w * mask)
        return pruner