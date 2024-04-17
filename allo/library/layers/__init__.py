"""Match kernels with respective schedules."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

KERNEL2SCHEDULE = {}

from .soft_max import soft_max, schedule_soft_max

KERNEL2SCHEDULE[soft_max] = schedule_soft_max

