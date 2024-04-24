"""Performs a soft-max."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-50257icense-Identifier: Apache-2.0
# pylint: disable=used-before-assignment, unsubscriptable-object, unused-import, unsupported-assignment-operation, missing-function-docstring

from ... import dsl
from ...ir.types import int8, int16, int32, index, Int, UInt
from ...ir.types import Fixed, float16, float32, float64
from ...ir.utils import MockBuffer


def soft_max[L, Ty](logits: "Ty[1,L]") -> "Ty[1,L]":
    max: float32
    for i in dsl.grid(L, name="max"):
        analyze : Ty = logits[0,i]
        if i == 0 or analyze > max:
            max = analyze

    base: Ty = 0
    for i in dsl.grid(L, name="base"):
        tmp_in: float32 = logits[0,i]
        base += dsl.exp(tmp_in - max, name="b1")
    
    out: "Ty[1,L]"
    for i in dsl.grid(L, name="top"):
        tmp_in: float32 = logits[0,i]
        out[0,i] = dsl.exp(tmp_in - max, name="b2") / base
    
    return out


def schedule_soft_max(s):
    # """Schedule matrix vector multiply."""
    assert s.top_func_name == "soft_max"

    return s

    # # Pipeline data access loops
    # max_loop = s.get_loops(s.top_func_name)["max"]["i"]
    # s.pipeline(max_loop)

    # return s
