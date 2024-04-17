import torch
import numpy as np
import allo
from allo import dsl
from allo.ir.types import index, int8, int16, int32, int64, int128, int256, int512
from allo.ir.types import uint8, uint16, uint32, uint64, uint128, uint256, uint512
from allo.ir.types import Fixed, float16, float32, float64
from allo.utils import get_np_struct_type
from allo.backend import hls


def soft_max_np(logits):
    np_logits = logits.numpy(force=True)[0]
    
    # toprint = np.array(np_logits)
    # toprint[toprint ==  np.inf] = 0
    # toprint[toprint == -np.inf] = 0
    # print(f"\n{min(toprint)}  {max(toprint)}")

    length = np_logits.shape[0]
    max_val = max(np_logits)

    base = 0
    for i in range(length):
        base += np.exp(np_logits[i] - max_val)

    func_out = np.zeros((1, length))
    for i in range(length):
        func_out[0,i] = np.exp(np_logits[i] - max_val) / base

    return torch.from_numpy(func_out)


def soft_max(logits):
    from allo.library.layers.soft_max import soft_max

    def top[L, Ty](logits_in: "Ty[1,L]") -> "Ty[1,L]":
        return soft_max(logits_in)

    np_logits = logits.numpy(force=True)[0]
    # print(f"\n{np_logits.shape[0]}\n")
    
    if 'soft_max_mod' not in globals():
        global soft_max_mod
        s_top = allo.customize(top, instantiate=[np_logits.shape[0], float32])
        soft_max_mod = s_top.build()

        s_top.compose(soft_max, instantiate=[np_logits.shape[0], float32])
        s_top.dataflow("top")

        hls_soft_max_mod = s_top.build(
            target="vitis_hls",
            mode="csim",
            project=f"soft_max_1.prj"
        )
        # csim_out = np.zeros(np_logits.shape)
        # hls_soft_max_mod(np_logits, csim_out)

    allo_out = soft_max_mod(np_logits)
    
    return torch.from_numpy(allo_out)
