from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from .io_utils import load_wav
from . import reconstruct, optimize, metrics

SR = 48000


def load_pair(di_path, target_path, sr=SR):
    x, _ = load_wav(di_path, sr); y, _ = load_wav(target_path, sr)
    return x.mean(0), y


def render(di, ir):
    """di: mono; ir: (C,N) -> (C, len(di))"""
    return np.stack([fftconvolve(di, h)[:len(di)] for h in ir])


def compare_channels(target, rendered, sr=SR):
    from .reconstruct import align_pair
    out = []
    for c in range(target.shape[0]):
        a, b, _ = align_pair(rendered[min(c, rendered.shape[0] - 1)], target[c], 0)
        out.append(metrics.compare(b, a, sr))
    return out
