"""Simulates a FIFO."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, missing-function-docstring

from .. import dsl
from ..ir.types import int8, int16, int32, index, Int, UInt
from ..ir.utils import MockBuffer

def fifo_init(list : "Ty[FIFO_L]", params : "index[3]"):
    params[0] = 0
    params[1] = 0
    params[2] = 0

def fifo_queue(list : "Ty[FIFO_L]", params : "index[3]", element : "Ty"):
    if params[2] == FIFO_L:
        asser_msg_fifo_full : "int8"
    params[2] += 1
    list[params[1]] = element
    params[1] = (params[1] + 1) % FIFO_L

def fifo_enqueue(list : "Ty[FIFO_L]", params : "index[3]") -> "Ty":
    if params[2] == 0:
        assert_read_when_empty : "int8"
    params[2] -= 1
    tmp : "Ty"
    tmp = list[params[0]]
    params[0] = (params[0] + 1) % FIFO_L
    return tmp

