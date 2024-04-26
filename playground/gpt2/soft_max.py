import sys
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

    length = np_logits.shape[0]
    max_val = max(np_logits)

    base = 0
    for i in range(length):
        base += np.exp(np_logits[i] - max_val)
    
    # print(f"\nnp base: {base}")

    func_out = np.zeros((1, length))
    for i in range(length):
        func_out[0,i] = np.exp(np_logits[i] - max_val) / base
        if func_out[0,i] != 0:
            print("{:.4e}, ".format(func_out[0,i]), end="")

    return torch.from_numpy(func_out)

csim = False
def soft_max(torch_logits):
    from allo.library.layers.soft_max import soft_max

    def top[L, Ty](logits_in: "Ty[1,L]") -> "Ty[1,L]":
        # top_output: "Ty[1, L]"
        top_output = soft_max(logits_in)
        return top_output

    np_logits = torch_logits.numpy(force=True)

    if 'soft_max_mod' not in globals():
        global soft_max_mod
        s_top = allo.customize(top, instantiate=[np_logits.shape[1], float32])
        soft_max_mod = s_top.build()
    mlir_out = soft_max_mod(np_logits)
    
    if csim:
        # if 'hls_soft_max_mod' not in globals():
        #     print("\ngenerating soft_max hls")
        #     global hls_soft_max_mod
        #     hls_s_top = allo.customize(top, instantiate=[np_logits.shape[1], float32])
        #     hls_s_top.compose(soft_max, instantiate=[np_logits.shape[1], float32])
        #     hls_soft_max_mod = hls_s_top.build(
        #         target="vitis_hls",
        #         mode="csim",
        #         project=f"soft_max_1.pr",
        #         # configs={
        #         #     "mappings": [
        #         #         None,
        #         #         None,
        #         #         None,
        #         #     ]
        #         # },
        #     )

        
        hls_soft_max_mod = allo.IPModule(
            top="top",
            headers=["soft_max_1.pr/kernel.h"],
            impls=["soft_max_1.pr/kernel.cpp"],
            signature=["float32[1]", "float32[1]"],
            link_hls=True,
        )

        csim_out = np.zeros(np_logits.shape[1])
        hls_soft_max_mod(np_logits)

    print()
    np_out = soft_max_np(torch_logits).numpy(force=True)    
    print()

    print(f"max: {np.max(np_out)}, min: {np.min(np_out)}")
    print(f"max: {np.max(csim_out)}, min: {np.min(csim_out)}")
    np.testing.assert_allclose(np_out, [csim_out], rtol=5e7)

    return torch.tensor(np.copy([csim_out]))
