"""IR export: 48 kHz / 24-bit PCM WAV, adjustable normalisation, never a hidden limiter."""
from __future__ import annotations
import os
import numpy as np
from .io_utils import resample, save_wav

DURATION_PRESETS_MS = {"Short (20 ms)": 20, "Medium (100 ms)": 100, "Long (200 ms)": 200}
EXPORT_SR = 48000


def trim_ir(ir, sr, ms, fade_ms=5.0):
    n = int(round(sr * ms / 1000)); ir = ir[:, :n]
    if ir.shape[1] < n:
        ir = np.pad(ir, ((0, 0), (0, n - ir.shape[1])))
    f = min(n // 2, max(8, int(sr * fade_ms / 1000)))
    ir = ir.copy(); ir[:, -f:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, f)))
    return ir


def normalize_ir(ir, peak_dbfs=-1.0, extra_gain_db=0.0):
    """Scales (one common gain for all channels => stereo image preserved) so peak = peak_dbfs + extra gain.
    If the result would exceed full scale it is scaled down instead of clipped. Returns (ir, warnings)."""
    pk = np.max(np.abs(ir)); warns = []
    if pk <= 0:
        raise ValueError("IR is silent")
    g = 10 ** ((peak_dbfs + extra_gain_db) / 20) / pk
    out = ir * g
    if np.max(np.abs(out)) > 0.9999:
        out *= 0.9999 / np.max(np.abs(out))
        warns.append("Requested gain exceeded 0 dBFS; reduced to avoid clipping (no limiter was applied).")
    return out, warns


def export_ir(path, ir, sr, duration_ms=None, peak_dbfs=-1.0, extra_gain_db=0.0, mono_only=False):
    ir = np.atleast_2d(ir)
    if mono_only and ir.shape[0] > 1:
        ir = ir.mean(0, keepdims=True)
    if sr != EXPORT_SR:
        ir = resample(ir, sr, EXPORT_SR)
    if duration_ms:
        ir = trim_ir(ir, EXPORT_SR, duration_ms)
    ir, warns = normalize_ir(ir, peak_dbfs, extra_gain_db)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_wav(path, ir, EXPORT_SR, "PCM_24")
    return warns


def export_many(folder, irs: dict, sr, duration_ms=None, **kw):
    log = {}
    for name, ir in irs.items():
        p = os.path.join(folder, f"{name}.wav"); log[name] = export_ir(p, ir, sr, duration_ms, **kw)
    return log
