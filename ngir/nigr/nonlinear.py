"""Nonlinear analysis (separate from IR reconstruction). All outputs are *indicators* (estimated).
An IR cannot represent any of this; nothing here is written into an IR."""
from __future__ import annotations
import os
import numpy as np
from scipy.signal import fftconvolve, csd, welch
from .reconstruct import align_pair, estimate_fir
from .io_utils import save_wav


def _rms_db(x): return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def analyze_nonlinear(x, y, sr, n_taps=4800):
    xa, ya, d = align_pair(x, y, 64)
    h, _, f, coh = estimate_fir([(x, y)], sr, n_taps, "freq", 1e-5, 64)
    yl = fftconvolve(xa, h)[:len(xa)]; ya = ya[:len(yl)]
    r = ya - yl
    out = {"delay_ms": d / sr * 1000}
    out["nonlinear_plus_noise_vs_output_db"] = _rms_db(r) - _rms_db(ya)
    mk = lambda lo, hi: (f >= lo) & (f < hi)
    out["coherence"] = {k: float(coh[mk(lo, hi)].mean()) for k, (lo, hi) in
                        {"80-400 Hz": (80, 400), "400-2k": (400, 2000), "2k-6k": (2000, 6000), "6k-12k": (6000, 12000)}.items()}
    # level-dependent gain (compression indicator)
    w = int(0.02 * sr); nfr = len(xa) // w
    xr = np.sqrt((xa[:nfr * w].reshape(nfr, w) ** 2).mean(1)); yr = np.sqrt((ya[:nfr * w].reshape(nfr, w) ** 2).mean(1))
    ok = xr > xr.max() * 10 ** (-50 / 20)
    if ok.sum() > 50:
        xl, g = 20 * np.log10(xr[ok]), 20 * np.log10(yr[ok] + 1e-12) - 20 * np.log10(xr[ok])
        q = np.quantile(xl, np.linspace(0, 1, 6)); curve = []
        for a, b in zip(q[:-1], q[1:]):
            s = (xl >= a) & (xl <= b); curve.append((float((a + b) / 2), float(g[s].mean())))
        out["gain_vs_input_level"] = curve
        out["gain_change_low_to_high_db"] = curve[-1][1] - curve[0][1]
        out["slope_db_per_db"] = float(np.polyfit(xl, g, 1)[0])  # <0 => compression/saturation
    # level-dependent tone: |H| from quiet vs loud segments
    seg = sr // 2; ns = len(xa) // seg
    if ns >= 8:
        lv = np.array([np.mean(xa[i * seg:(i + 1) * seg] ** 2) for i in range(ns)]); o = np.argsort(lv); k = max(2, ns // 3)
        def Hmag(ids):
            xs = np.concatenate([xa[i * seg:(i + 1) * seg] for i in ids]); ys = np.concatenate([ya[i * seg:(i + 1) * seg] for i in ids])
            ff, pxx = welch(xs, sr, nperseg=4096); _, pxy = csd(xs, ys, sr, nperseg=4096); return ff, 20 * np.log10(np.abs(pxy / (pxx + 1e-20)) + 1e-12)
        ff, hq = Hmag(o[:k]); _, hl = Hmag(o[-k:])
        out["tone_change_loud_minus_quiet_db"] = {n: float(np.mean((hl - hq)[(ff >= lo) & (ff < hi)]))
                                                  for n, (lo, hi) in {"<300 Hz": (80, 300), "300-3k": (300, 3000), ">3k": (3000, 10000)}.items()}
    # clipping shape heuristic
    ay = np.abs(ya); pk = ay.max()
    out["crest_in_db"] = float(20 * np.log10(np.abs(xa).max() / (np.sqrt(np.mean(xa ** 2)) + 1e-20)))
    out["crest_out_db"] = float(20 * np.log10(pk / (np.sqrt(np.mean(ya ** 2)) + 1e-20)))
    frac = float(np.mean(ay > 0.97 * pk))
    out["fraction_near_peak"] = frac
    out["clip_character"] = ("hard-clipping likely" if frac > 0.01 else "soft saturation / no hard clipping evident") + " (heuristic)"
    out["verdict"] = ("Strongly nonlinear: an IR alone cannot reproduce this; consider NAM capture." if out["nonlinear_plus_noise_vs_output_db"] > -12
                      else "Mostly linear: an IR can represent most of the signal chain." if out["nonlinear_plus_noise_vs_output_db"] < -25
                      else "Mixed: IR captures the linear part; nonlinear residual remains.")
    return out


def export_nam_pair(folder, x, y, sr, name="pair"):
    """Writes time-aligned mono 24-bit input/output WAVs for NAM-style training.
    NOTE: the official NAM trainer expects its own calibrated 'input.wav' reamp signal; arbitrary DI
    material is only usable in advanced/custom training setups. See README."""
    os.makedirs(folder, exist_ok=True)
    xa, ya, _ = align_pair(x, y, 0); n = min(len(xa), len(ya))
    save_wav(os.path.join(folder, f"{name}_input_aligned.wav"), xa[:n] / max(1, np.abs(xa).max()), sr)
    save_wav(os.path.join(folder, f"{name}_output_aligned.wav"), ya[:n] / max(1, np.abs(ya).max()), sr)
    return folder


def combined_preview(x, h, drive=1.0, shape="tanh"):
    """Combined chain approximation PREVIEW: static waveshaper -> linear IR. Heuristic, not a trained model."""
    s = np.tanh(drive * x) if shape == "tanh" else np.clip(drive * x, -1, 1)
    return fftconvolve(s, h)[:len(x)]
