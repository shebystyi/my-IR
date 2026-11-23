"""Per-recording analysis. Every value is tagged 'measured' or 'estimated'."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, group_delay
from . import metrics
from .reconstruct import smooth_octave

BANDS = {"sub 20-80": (20, 80), "low 80-250": (80, 250), "low-mid 250-500": (250, 500),
         "mid 500-2k": (500, 2000), "high-mid 2k-5k": (2000, 5000), "presence 5k-8k": (5000, 8000),
         "air 8k-16k": (8000, 16000)}


def _db(x): return 20 * np.log10(np.maximum(x, 1e-12))


def envelope(x, sr, ms=2.0):
    w = max(1, int(sr * ms / 1000)); return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, "same"))


def detect_f0(x, sr, fmin=70, fmax=1000):
    n = int(0.25 * sr); i = int(np.argmax(envelope(x, sr, 20))); s = max(0, min(i, len(x) - n))
    seg = x[s:s + n] * np.hanning(min(n, len(x)))
    if len(seg) < sr // 20: return None, 0.0
    ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * len(seg))) ** 2)[:len(seg)]
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi >= len(ac): return None, 0.0
    k = lo + int(np.argmax(ac[lo:hi])); conf = ac[k] / (ac[0] + 1e-20)
    return sr / k, float(conf)


def analyze(x, sr):
    """x: (C,N). Returns dict(metrics=[(name,value,unit,kind)], ltas=(f,p_db), env=..., notes=[...])."""
    x = np.atleast_2d(x); m = x.mean(0); M = []
    add = lambda n, v, u="", k="measured": M.append((n, v, u, k))
    pk = np.max(np.abs(x)); rms = np.sqrt(np.mean(m ** 2))
    add("Peak", float(_db(pk)), "dBFS"); add("RMS", float(_db(rms)), "dBFS")
    add("Crest factor", float(_db(pk) - _db(rms)), "dB")
    w = int(0.05 * sr); fr = np.sqrt(np.convolve(m ** 2, np.ones(w) / w, "valid")[::w // 2] + 1e-20)
    act = fr[_db(fr) > _db(fr).max() - 60]
    add("Dynamic range (p95-p10 of 50 ms RMS)", float(np.percentile(_db(act), 95) - np.percentile(_db(act), 10)), "dB")
    flat = np.sum(np.abs(m) >= 0.999 * pk) if pk > 0.98 else 0
    add("Samples at full scale", int(flat), "", "measured")
    add("Clipping detected", bool(flat >= 3), "", "measured")
    # envelope-based dynamics (material dependent => estimated)
    env = envelope(m, sr, 2.0); ip = int(np.argmax(env)); e = env[ip]
    pre = env[:ip + 1]
    try:
        t10 = np.where(pre >= 0.1 * e)[0][0]; t90 = np.where(pre >= 0.9 * e)[0][0]
        add("Attack time (10-90 %)", float((t90 - t10) / sr * 1000), "ms", "estimated")
    except IndexError: pass
    post = _db(env[ip:ip + int(3 * sr)]); t = np.arange(len(post)) / sr
    if len(post) > sr // 10:
        sel = post > post.max() - 25
        if sel.sum() > 20:
            add("Decay rate", float(np.polyfit(t[sel], post[sel], 1)[0]), "dB/s", "estimated")
        j = min(int(0.5 * sr), len(post) - 1); add("Sustain level @500 ms", float(post[j] - post[0]), "dB", "estimated")
    f, p = metrics.ltas(m, sr); pd = 10 * np.log10(p + 1e-20); tot = p.sum()
    for n, (lo, hi) in BANDS.items():
        mk = (f >= lo) & (f < hi); add(f"Band {n}", float(10 * np.log10(p[mk].sum() / tot + 1e-20)), "dB rel. total")
    cum = np.cumsum(p) / tot
    add("Spectral centroid", float((f * p).sum() / tot), "Hz"); add("Rolloff 95 %", float(f[np.searchsorted(cum, .95)]), "Hz")
    sm = 10 * np.log10(smooth_octave(f, p, 6) + 1e-20); mid = sm[(f > 500) & (f < 2000)].mean()
    idx = np.where((f > 1000) & (sm < mid - 30))[0]
    add("HF roll-off (-30 dB vs mid)", float(f[idx[0]]) if len(idx) else float(f[-1]), "Hz", "estimated")
    d = sm - (np.convolve(np.pad(sm, 30, mode="edge"), np.ones(61) / 61, "valid"))
    band = (f > 60) & (f < 12000)
    pkk, _ = find_peaks(np.where(band, d, -99), prominence=3, distance=20); nt, _ = find_peaks(np.where(band, -d, -99), prominence=3, distance=20)
    pr = sorted(pkk, key=lambda i: -d[i])[:5]; nr = sorted(nt, key=lambda i: d[i])[:5]
    add("Resonant peaks", ", ".join(f"{f[i]:.0f} Hz ({d[i]:+.1f} dB)" for i in pr) or "-", "", "estimated")
    add("Notches", ", ".join(f"{f[i]:.0f} Hz ({d[i]:+.1f} dB)" for i in nr) or "-", "", "estimated")
    f0, conf = detect_f0(m, sr)
    if f0 and conf > 0.6:
        add("Fundamental", float(f0), "Hz", "estimated")
        n = 1 << int(np.ceil(np.log2(sr))); i0 = int(np.argmax(envelope(m, sr, 20))); s = max(0, min(i0, len(m) - n)); seg = m[s:s + n]
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)); fr_ = np.fft.rfftfreq(len(seg) if len(seg) == n else n, 1 / sr)
        hk = []
        for k in range(1, 9):
            c = int(round(k * f0 * n / sr))
            if c + 3 < len(S): hk.append(S[c - 3:c + 4].max())
        if len(hk) >= 4:
            hk = np.array(hk); add("Harmonics H1..H8", ", ".join(f"{v:.1f}" for v in _db(hk / hk[0])), "dB rel. H1", "estimated")
            od, ev = np.sum(hk[2::2] ** 2), np.sum(hk[1::2] ** 2)
            add("Odd/even harmonic balance (H3.. vs H2..)", float(10 * np.log10(od / (ev + 1e-20))), "dB", "estimated")
            add("THD estimate (H2-H8, single-note only)", float(100 * np.sqrt(np.sum(hk[1:] ** 2) / hk[0] ** 2)), "%", "estimated")
    else:
        add("Fundamental / harmonics", "n/a (polyphonic or noisy material)", "", "estimated")
    if x.shape[0] == 2:
        a, b = x; mid, side = (a + b) / 2, (a - b) / 2
        add("Channel correlation", float(np.corrcoef(a, b)[0, 1]), "", "measured")
        add("Stereo width (S/M energy)", float(10 * np.log10((np.sum(side ** 2) + 1e-20) / (np.sum(mid ** 2) + 1e-20))), "dB", "measured")
    return {"metrics": M, "ltas": (f, pd), "envelope": env, "sr": sr}


def ir_report(h, sr):
    """Measured properties of an (estimated) IR."""
    h = np.asarray(h); e = h ** 2; sch = 10 * np.log10(np.cumsum(e[::-1])[::-1] / e.sum() + 1e-20)
    t60 = np.where(sch < -30)[0]
    w, gd = group_delay((h, [1.0]), w=4096, fs=sr)
    band = (w > 200) & (w < 5000)
    early = e[: int(0.005 * sr) + int(np.argmax(np.abs(h)))].sum() / e.sum()
    return [("Peak position", float(np.argmax(np.abs(h)) / sr * 1000), "ms", "measured"),
            ("Decay to -30 dB (Schroeder)", float(t60[0] / sr * 1000) if len(t60) else float(len(h) / sr * 1000), "ms", "measured"),
            ("Mean group delay 200 Hz-5 kHz", float(np.mean(gd[band]) / sr * 1000), "ms", "measured"),
            ("Energy in first 5 ms after peak", float(100 * early), "%", "measured"),
            ("Room/early-reflection content", "not separable from cabinet in a single estimated IR", "", "estimated")]


def combined_profile(results):
    """Average long-term spectra of several analyses (equal weight)."""
    f = results[0]["ltas"][0]; P = np.mean([r["ltas"][1] for r in results], axis=0)
    return f, P
