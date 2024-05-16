"""Runs a set of tests on
an allo module."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, unrecognized-inline-option, import-outside-toplevel, too-many-locals

import multiprocessing as mp
import numpy as np
import allo
from allo.backend import hls

def run_test(function, sub_module, reference, top_config, config, input, runs=(1, 0, False)):
    """Runs test configurations for given allo module."""
    np.set_printoptions(formatter={"int": hex})

    mlir_runs, csim_runs, compile = runs
    
    s_top = allo.customize(function, instantiate=[*top_config])

    # CPU testing
    if callable(input):
        data_in = [input(config) for _ in range(mlir_runs)]
    else:
        data_in = input

    mod = s_top.build()

    out_allo = [mod(*data_in_line) for data_in_line in data_in]
    out_ref = [reference(*data_in_line) for data_in_line in data_in]
    np.testing.assert_equal(out_allo, out_ref)
    print(f"Passed! {config}")

    # confirm vitis usage
    if csim_runs == 0 and not compile:
        return
    assert hls.is_available("vitis_hls"), "Vitis HLS not found"

    # Compose with submodule
    s_top.compose(sub_module, instantiate=[*config])

    if csim_runs != 0:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="csim",
            project=f"{function.__name__}_{config}_csim.prj",
        )

        out_csim = np.zeros((csim_runs, *out_ref.shape[1:]))
        for data_in_itr, out_csim_itr in zip(data_in, out_csim):
            hls_mod(*data_in_itr, out_csim_itr)

        np.testing.assert_equal(out_csim, out_ref[0 : csim_runs])
        print(f"Passed csim! {config}")

    if compile:
        hls_mod = s_top.build(
            target="vitis_hls",
            mode="hw",
            project=f"{function.__name__}_{config}.prj",
        )
        hls_mod()

def init_run_tests(function, sub_module, reference, tests):
    """Runs series of tests on given allo module."""

    processes = [
        mp.Process(target=run_test, args=(function, sub_module, reference, *args)) for args in tests
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()