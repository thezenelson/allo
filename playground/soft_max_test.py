"""Runs a comprehensive set of tests on
the matrix vector multiply module."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, unrecognized-inline-option, import-outside-toplevel, too-many-locals

import os
import sys
import multiprocessing as mp
import numpy as np
import torch
import allo
from allo import dsl
from allo.ir.types import index, int8, int16, int32, int64, int128, int256, int512
from allo.ir.types import uint8, uint16, uint32, uint64, uint128, uint256, uint512
from allo.ir.types import Fixed, float16, float32, float64
from allo.backend import hls

T, F = True, False

# Test setup for soft max multiply
def test_basic_soft_max(runs=10, hw_runs=0, ll=2):
    """Runs selected test configuration of mat vec multiply."""
    from allo.library.layers.soft_max import soft_max

    # np.random.seed(seed=400)
    np.set_printoptions(formatter={"int": hex})

    def top[L, Ty](logits_in: "Ty[1,L]") -> "Ty[1,L]":
        # top_output: "Ty[1, L]"
        top_output = soft_max(logits_in)
        return top_output
        

    s_top = allo.customize(top, instantiate=[ll, float32])

    # CPU testing
    mod = s_top.build()

    x = np.random.random_sample(size=(runs, 1, ll)).astype(np.float32)

    c_allo = [mod(x_itr) for x_itr in x]
    c_torch = [torch.nn.functional.softmax(torch.from_numpy(x_itr), dim=-1).numpy(force=True) for x_itr in x]

    np.testing.assert_allclose(c_allo, c_torch, rtol=1e-05, err_msg=f"mlir {ll}")
    print(f"Passed! {ll}")

    # confirm vitis usage
    if hw_runs == 0:
        return
    assert hls.is_available("vitis_hls"), "Vitis HLS not found"

    # Compose with submodule
    s_top = allo.customize(top, instantiate=[ll, float32])
    s_top.compose(soft_max, instantiate=[ll, float32])

    hls_mod = s_top.build(
        target="vitis_hls",
        mode="csim",
        project=f"soft_max_{ll}_csim.prj",
    )

    x = np.random.random_sample(size=(runs, 1, ll)).astype(np.float32)

    c_csim = np.zeros((hw_runs, 1, ll), dtype=np.float32)
    for x_itr, c_csim_itr in zip(x, c_csim):
        hls_mod(x_itr, c_csim_itr)

    np.testing.assert_allclose(c_csim, c_torch[0 : hw_runs], rtol=1e-05, err_msg=f"csim {ll}")
    print(f"Passed csim! {ll}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # All full tests
    # fmt: off
    full_tests = [
      # (runs, hw_r, L,     ),
        (100,  1,    10,    ),
        (100,  F,    1000,  ),
        (100,  F,    10000, ),
        (100,  F,    50000, ),
    ]
    # fmt: on

    # Run each test in it's own process
    processes = [
        mp.Process(target=test_basic_soft_max, args=args) for args in full_tests
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
