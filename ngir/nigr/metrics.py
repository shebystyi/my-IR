"""Objective comparison metrics. No single 'similarity %' is produced on purpose."""
from __future__ import annotations
import numpy as np
from scipy.signal import stft, welch

BANDS = [(40, 120), (120, 400), (400, 1600), (1600, 5000), (5000, 12000)]


def _mag(x, n):
    _, _, Z = stft(x, nperseg=n, noverlap=n - n // 4, window="hann", boundary=None, padded=False)
    return np.abs(Z)


def mr_stft(ref, est, sizes=(512, 1024, 2048)):
    sc, lm = [], []
    for n in sizes:
        if len(ref) < n or len(est) < n:
            continue
        A, B = _mag(ref, n), _mag(est, n)
        sc.append(np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-12))
        lm.append(np.mean(np.abs(np.log(A + 1e-7) - np.log(B + 1e-7))))
    return float(np.mean(sc)), float(np.mean(lm))


def ltas(x, sr, nperseg=8192):
    f, p = welch(x, sr, nperseg=min(nperseg, len(x)), noverlap=None, detrend=False)
    return f, p


def log_spectral_distance(ref, est, sr, fmin=40, fmax=16000):
    f, p1 = ltas(ref, sr); _, p2 = ltas(est, sr)
    m = (f >= fmin) & (f <= fmax)
    d = 10 * np.log10(p1[m] + 1e-20) - 10 * np.log10(p2[m] + 1e-20)
    return float(np.sqrt(np.mean(d ** 2)))


def band_errors_db(ref, est, sr):
    f, p1 = ltas(ref, sr); _, p2 = ltas(est, sr)
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"{lo}-{hi} Hz"] = float(10 * np.log10((p2[m].sum() + 1e-20) / (p1[m].sum() + 1e-20)))
    return out


def time_error_db(ref, est):
    n = min(len(ref), len(est)); r, e = ref[:n], est[:n]
    g = np.dot(r, e) / (np.dot(e, e) + 1e-20)
    return float(10 * np.log10(np.sum((r - g * e) ** 2) / (np.sum(r ** 2) + 1e-20) + 1e-20))


def envelope_db(x, sr, win_ms=5.0):
    w = max(1, int(sr * win_ms / 1000))
    e = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same") + 1e-20)
    return 20 * np.log10(e)


def envelope_diff_db(ref, est, sr):
    n = min(len(ref), len(est))
    a, b = envelope_db(ref[:n], sr), envelope_db(est[:n], sr)
    m = a > (a.max() - 50)
    return float(np.mean(np.abs(a[m] - b[m])))


def match_loudness(ref, est):
    g = np.sqrt(np.mean(ref ** 2) / (np.mean(est ** 2) + 1e-20))
    return est * g, float(20 * np.log10(g))


def compare(ref, est, sr):
    """ref/est mono 1-D arrays, already time-aligned. est is loudness (RMS) matched first."""
    n = min(len(ref), len(est)); ref, est = ref[:n], est[:n]
    est, gain = match_loudness(ref, est)
    sc, lm = mr_stft(ref, est)
    return {
        "loudness_match_gain_db": gain,
        "log_spectral_distance_db": log_spectral_distance(ref, est, sr),
        "mr_stft_spectral_convergence": sc,
        "mr_stft_log_mag_l1": lm,
        "time_domain_nmse_db": time_error_db(ref, est),
        "envelope_diff_db": envelope_diff_db(ref, est, sr),
        "band_error_db": band_errors_db(ref, est, sr),
    }
