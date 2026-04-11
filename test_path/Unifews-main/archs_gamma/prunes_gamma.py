import os
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from tensorlayerx.nn import Module

# ===================== 【HELPERS】Internal Math Helpers =====================
# 🔥 修复1：兼容TLX维度规范，axis支持int/tuple，避免SAGEConv维度报错
def safe_norm(x, p=2, axis=None, keepdims=False):
    """Replacement for tlx.norm which is missing in some TLX versions"""
    x_abs = tlx.abs(x)
    if p == 1:
        return tlx.reduce_sum(x_abs, axis=axis, keepdims=keepdims)
    # Standard L2 norm
    # 🔥 原：tlx.clamp_min → 替换为 TLX 通用 clip_by_value
    return tlx.sqrt(tlx.clip_by_value(tlx.reduce_sum(x_abs * x_abs, axis=axis, keepdims=keepdims), clip_value_min=1e-9, clip_value_max=float('inf')))

# ===================== 【新增】剪枝基类 & 模拟 PyTorch prune 工具 =====================
class BasePruningMethod:
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask=None):
        raise NotImplementedError("子类必须实现 compute_mask")

    @classmethod
    def apply(cls, module, name, *args, **kwargs):
        return cls(*args, **kwargs)

class prune:
    @staticmethod
    def is_pruned(module: Module) -> bool:
        return any(key.endswith("_mask") for key in dir(module))

    @staticmethod
    def remove(module: Module, name: str):
        # 🔥 修复2：统一调用rewind，无冗余逻辑，避免参数名混淆
        rewind(module, name)

# ===================== 原有剪枝工具函数 =====================
def prune_threshold(x, threshold=1e-3):
    norm_vals = safe_norm(x, axis=1) / x.shape[1]
    idx_0 = norm_vals < threshold
    x = tlx.where(idx_0, tlx.zeros_like(x), x)
    return x, idx_0

def prune_topk(x, k=0.2):
    num_0 = int(x.shape[0] * k)
    x_norm = safe_norm(x, axis=1)
    _, idx_0 = tlx.topk(x_norm, num_0)
    mask_val = tlx.ones((x.shape[0],), dtype=tlx.bool)
    mask = tlx.convert_to_tensor(tlx.convert_to_numpy(mask_val))
    mask = tlx.where(tlx.arange(x.shape[0])[:, None] != idx_0, True, False)
    mask = tlx.reduce_any(mask, axis=1)
    x = tlx.where(mask[:, None], x, tlx.zeros_like(x))
    return x, idx_0

def rewind(module: Module, name: str):
    """恢复剪枝参数：TLX-torch后端专用，用.data赋值"""
    orig_name = name + "_orig"
    if hasattr(module, orig_name):
        weight_param = getattr(module, name)
        orig_data = getattr(module, orig_name)
        
        weight_param.data = orig_data
        
        delattr(module, orig_name)
        if hasattr(module, name + "_mask"):
            delattr(module, name + "_mask")

# ===================== 剪枝类：彻底解决Parameter类型错误 =====================
class ThrInPrune(BasePruningMethod):
    PRUNING_TYPE = 'structured'

    def __init__(self, threshold, dim=0):
        self.threshold = threshold
        self.dim = dim

    def compute_mask(self, t):
        t_abs = tlx.abs(t)
        tmax = tlx.reduce_max(t_abs)
        # 🔥 修复3：原 tlx.clamp_min → 替换为 TLX clip_by_value
        tmax = tlx.clip_by_value(tmax, clip_value_min=1e-9, clip_value_max=float('inf'))
        # 🔥 修复4：原 tlx.clamp → 替换为 TLX clip_by_value
        thresh = tlx.clip_by_value(self.threshold, clip_value_min=1e-9, clip_value_max=tmax)
        mask = tlx.where(t_abs < thresh, 0.0, 1.0)
        return mask

    @classmethod
    def apply(cls, module: Module, name: str, threshold):
        pruner = cls(threshold)
        weight = getattr(module, name)
        mask = pruner.compute_mask(weight)

        orig_name = name + "_orig"
        if not hasattr(module, orig_name):
            setattr(module, orig_name, weight.data.clone())

        weight.data = weight.data * mask
        
        setattr(module, name + "_mask", mask)
        return pruner

class ThrProdPrune(BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, t, importance_scores=None):
        if importance_scores is not None:
            t = importance_scores
        mask = tlx.where(tlx.abs(t) < self.threshold, 0.0, 1.0)
        return mask

    @classmethod
    def apply(cls, module: Module, name: str, threshold, x):
        w = getattr(module, name)
        score = tlx.abs(w) * safe_norm(x, axis=0)
        pruner = cls(threshold)
        mask = pruner.compute_mask(w, importance_scores=score)

        orig_name = name + "_orig"
        if not hasattr(module, orig_name):
            setattr(module, orig_name, w.data.clone())

        w.data = w.data * mask
        
        setattr(module, name + "_mask", mask)
        return pruner