"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os
import random

import numpy as np
import jittor as jt


def exists(x):
    return x is not None


def set_seed(seed: int = 0):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model, include_buffers: bool = False):
    n_trainable_params = sum(p.numel() for p in model.parameters())

    return n_trainable_params


def sort_dict(D: dict):
    s_keys = sorted(D.keys())
    return {k: D[k] for k in s_keys}
