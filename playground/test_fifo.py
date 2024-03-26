"""Test running a fifo in mlir."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, unrecognized-inline-option, import-outside-toplevel, too-many-locals

import os
import sys
import multiprocessing as mp
import numpy as np
import allo
from allo import dsl
from allo.ir.types import index, int8, int16, int32, int64, int128, int256, int512, Struct
from allo.ir.types import uint8, uint16, uint32, uint64, uint128, uint256, uint512
from allo.utils import get_np_struct_type
from allo.backend import hls

def fifo_1():
    """First test of fifo module."""

    from allo.library.fifo import fifo_init, fifo_queue, fifo_enqueue

    def top[FIFO_L, Ty]() -> "Ty[10]":
        returns : "Ty[10]"
        
        test_list : "Ty[FIFO_L]"
        test_params : "index[3]"    # start, end
        fifo_init(test_list, test_params)

        fifo_queue(test_list, test_params, 10)
        fifo_queue(test_list, test_params, 20)
        fifo_queue(test_list, test_params, 30)
        fifo_queue(test_list, test_params, 40)
        fifo_queue(test_list, test_params, 24)
        returns[0] = fifo_enqueue(test_list, test_params)
        returns[1] = fifo_enqueue(test_list, test_params)
        returns[2] = fifo_enqueue(test_list, test_params)
        returns[3] = fifo_enqueue(test_list, test_params)
        returns[4] = fifo_enqueue(test_list, test_params)
        
        fifo_queue(test_list, test_params, 10)
        fifo_queue(test_list, test_params, 20)
        fifo_queue(test_list, test_params, 30)
        fifo_queue(test_list, test_params, 40)
        fifo_queue(test_list, test_params, 24)
        returns[5] = fifo_enqueue(test_list, test_params)
        returns[6] = fifo_enqueue(test_list, test_params)
        returns[7] = fifo_enqueue(test_list, test_params)
        returns[8] = fifo_enqueue(test_list, test_params)
        returns[9] = fifo_enqueue(test_list, test_params)

        # test_struct : "Struct['brand': 'Ty', 'model': 'Ty']"
        return returns

    s_top = allo.customize(top, instantiate=[7, int32])
    mod = s_top.build()
    print(mod())

if __name__ == "__main__":
    fifo_1()