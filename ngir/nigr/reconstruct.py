"""Linear (LTI) FIR reconstruction.

Paired mode (DI x -> processed y): regularised least-squares estimate of h in y ~= h * x,
  * 'freq': Tikhonov/Wiener estimator  H = Sxy / (Sxx + lambda(f))  (Welch-averaged over all pairs)
  * 'time': normal equations (Toeplitz, ridge-regularised) solved with Levinson recursion
Coherence is reported per frequency: where it is low, the linear model is NOT valid
(noise or nonlinear distortion) and the IR there is uncertain.

Approximate mode (no DI): spectral-match minimum-phase IR. NOT a physically recovered IR."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import welch, csd, fftconvolve
from scipy.linalg import solve_toeplitz

from . import metrics


@dataclass
class IRResult:
    ir: np.ndarray                 # (channels, taps)
    sr: int
    method: str
    physical: bool                 # True only for paired reconstruction
    coherence: tuple | None = None  # (freqs, coherence[ch][f])
    delay_samples: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def estimate_delay(x, y, max_lag):
    n = min(len(x), len(y)); nfft = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y[:n], nfft) * np.conj(np.fft.rfft(x[:n], nfft)), nfft)
    cc = np.concatenate([c[-max_lag:], c[:max_lag + 1]])
    return int(np.argmax(np.abs(cc)) - max_lag)


def align_pair(x, y, pre=64, max_lag=None):
    """Delay/advance x so that the dominant IR peak sits near index `pre` (keeps pre-ringing)."""
    max_lag = max_lag or min(len(x), len(y)) // 4
    d = estimate_delay(x, y, max_lag)
    s = d - pre
    xa = np.concatenate([np.zeros(s), x]) if s >= 0 else x[-s:]
    n = min(len(xa), len(y))
    return xa[:n], y[:n], d


def _fade_tail(h, frac=0.1):
    n = max(8, int(len(h) * frac))
    h = h.copy(); h[-n:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, n)))
    return h


def _reg_curve(freqs, floor, hf_boost):
    """Frequency-dependent regularisation weight: 1 in-band, rising above 12k and below 40 Hz."""
    w = np.ones_like(freqs)
    w[freqs < 40] *= 10.0
    w[freqs > 12000] *= (1.0 + hf_boost)
    return w * floor


def estimate_fir(pairs, sr, n_taps, method="freq", reg=1e-5, pre=64, hf_reg_boost=5.0):
    """pairs: list of (x, y) 1-D arrays at sample rate `sr`. Returns (h, delay, freqs, coherence)."""
    al = [align_pair(x, y, pre) for x, y in pairs]
    if method == "time":
        rxx = np.zeros(n_taps); rxy = np.zeros(n_taps)
        for xa, ya, _ in al:
            nfft = 1 << int(np.ceil(np.log2(len(xa) + n_taps)))
            X, Y = np.fft.rfft(xa, nfft), np.fft.rfft(ya, nfft)
            rxx += np.fft.irfft(np.abs(X) ** 2, nfft)[:n_taps]
            rxy += np.fft.irfft(Y * np.conj(X), nfft)[:n_taps]
        ridge = np.zeros(n_taps); ridge[0] = reg * rxx[0] * 10  # white ridge
        h = solve_toeplitz(rxx + ridge, rxy)
        nseg = 1 << int(np.ceil(np.log2(4 * n_taps)))
    else:
        nseg = 1 << int(np.ceil(np.log2(4 * n_taps)))
        Pxx = Pxy = Pyy = 0; tot = 0
        for xa, ya, _ in al:
            if len(xa) < nseg:
                xa = np.pad(xa, (0, nseg - len(xa))); ya = np.pad(ya, (0, nseg - len(ya)))
            f, pxx = welch(xa, sr, nperseg=nseg, detrend=False, scaling="spectrum")
            _, pxy = csd(xa, ya, sr, nperseg=nseg, detrend=False, scaling="spectrum")
            _, pyy = welch(ya, sr, nperseg=nseg, detrend=False, scaling="spectrum")
            w = len(xa); tot += w
            Pxx = Pxx + w * pxx; Pxy = Pxy + w * pxy; Pyy = Pyy + w * pyy
        Pxx, Pxy, Pyy = Pxx / tot, Pxy / tot, Pyy / tot
        lam = _reg_curve(f, reg * np.mean(Pxx), hf_reg_boost)
        H = Pxy / (Pxx + lam)
        h = np.fft.irfft(H, nseg)[:n_taps]
    # coherence (always from Welch, for uncertainty reporting)
    Pxx = Pxy = Pyy = 0; tot = 0
    for xa, ya, _ in al:
        if len(xa) < nseg:
            xa = np.pad(xa, (0, nseg - len(xa))); ya = np.pad(ya, (0, nseg - len(ya)))
        f, pxx = welch(xa, sr, nperseg=nseg, detrend=False)
        _, pxy = csd(xa, ya, sr, nperseg=nseg, detrend=False)
        _, pyy = welch(ya, sr, nperseg=nseg, detrend=False)
        Pxx = Pxx + len(xa) * pxx; Pxy = Pxy + len(xa) * pxy; Pyy = Pyy + len(xa) * pyy; tot += len(xa)
    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-30)
    return _fade_tail(h), [a[2] for a in al], f, np.clip(coh, 0, 1)


def reconstruct_paired(pairs, sr, n_taps=4800, method="freq", reg=1e-5, pre=64, auto_reg=False):
    """pairs: list of (x_mono, y[C,N]). Builds one IR channel per target channel."""
    C = max(y.shape[0] for _, y in pairs)
    warns, irs, cohs, delays = [], [], [], []
    for c in range(C):
        pc = [(x, y[min(c, y.shape[0] - 1)]) for x, y in pairs]
        r = reg
        if auto_reg:
            r = select_reg(pc, sr, n_taps, method, pre)
        h, d, f, coh = estimate_fir(pc, sr, n_taps, method, r, pre)
        irs.append(h); cohs.append(coh); delays += d
    freqs = f
    ir = np.stack(irs)
    mc = np.mean([np.mean(c[(freqs > 80) & (freqs < 8000)]) for c in cohs])
    if mc < 0.9:
        warns.append(f"Mean coherence 80 Hz-8 kHz is {mc:.2f}: the target contains significant nonlinear "
                     "distortion and/or noise; an LTI IR can only represent the linear part.")
    if len(pairs[0][0]) < sr * 5:
        warns.append("Short material: the estimate may be poorly conditioned.")
    return IRResult(ir, sr, f"paired-{method}", True, (freqs, np.stack(cohs)), delays, warns)


def select_reg(pairs, sr, n_taps, method, pre, grid=(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)):
    """Pick the regularisation with lowest MR-STFT error on a held-out 25 % of each recording."""
    train = [(x[:int(len(x) * .75)], y[:int(len(y) * .75)]) for x, y in pairs]
    best, best_e = grid[2], np.inf
    for g in grid:
        h, *_ = estimate_fir(train, sr, n_taps, method, g, pre)
        e = 0
        for x, y in pairs:
            xa, ya, _ = align_pair(x, y, pre)
            k = int(len(xa) * .75)
            est = fftconvolve(xa, h)[:len(xa)]
            e += metrics.mr_stft(ya[k:], est[k:])[0]
        if e < best_e:
            best, best_e = g, e
    return best


# ----------------------------------------------------------------- approximate mode
def min_phase_from_mag(mag, nfft):
    c = np.fft.irfft(np.log(np.maximum(mag, 1e-9)), nfft)
    w = np.zeros(nfft); w[0] = 1; w[1:nfft // 2] = 2; w[nfft // 2] = 1
    return np.fft.irfft(np.exp(np.fft.rfft(c * w)), nfft)


def smooth_octave(freqs, mag, frac=6):
    lf = np.log2(np.maximum(freqs, 1.0)); grid = np.linspace(lf[1], lf[-1], 2048)
    m = np.interp(grid, lf[1:], 10 * np.log10(mag[1:] + 1e-20))
    k = max(1, int((1 / frac) / (grid[1] - grid[0])))
    m = np.convolve(np.pad(m, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return 10 ** (np.interp(lf, grid, m) / 10)


def reconstruct_approximate(ref_signals, sr, n_taps=4800, source_signals=None, smooth_frac=6):
    """ref_signals: list of mono arrays (finished recordings). source_signals: optional mono arrays of
    YOUR guitar through Neural DSP with a flat/no IR - strongly recommended.
    Result = minimum-phase filter matching ratio of long-term spectra. NOT a recovered IR."""
    warns = ["APPROXIMATE MODE: spectral match only. The result is NOT a physically recovered impulse "
             "response; amp distortion, compression, mastering EQ and the source guitar's spectrum are folded in."]
    nfft = 1 << int(np.ceil(np.log2(max(n_taps * 4, 16384))))
    def avg(sigs):
        P = 0
        for s in sigs:
            f, p = metrics.ltas(s, sr, nperseg=nfft // 2 if len(s) >= nfft // 2 else len(s))
            P = P + np.interp(np.linspace(0, sr / 2, nfft // 2 + 1), f, p) / (np.mean(p) + 1e-20)
        return P / len(sigs)
    fr = np.linspace(0, sr / 2, nfft // 2 + 1)
    Pr = avg(ref_signals)
    if source_signals:
        Ps = avg(source_signals)
    else:
        Ps = 1.0 / np.maximum(fr, 50.0) ** 1.5  # generic guess: this is an ASSUMPTION
        Ps = Ps / Ps[np.searchsorted(fr, 1000)] * Pr[np.searchsorted(fr, 1000)]
        warns.append("No source (your-guitar-through-plugin) recordings given: a generic spectral tilt was "
                     "ASSUMED for the source. Accuracy is low; supply source recordings for a better match.")
    ratio = smooth_octave(fr, Pr / (Ps + 1e-20), smooth_frac)
    mag = np.sqrt(ratio)
    mag[fr < 60] *= np.clip(fr[fr < 60] / 60, 0.05, 1) ** 2          # high-pass
    mag[fr > 17000] *= np.clip((20000 - fr[fr > 17000]) / 3000, 0.01, 1)  # low-pass
    h = min_phase_from_mag(mag, nfft)[:n_taps]
    h = _fade_tail(h)
    return IRResult(h[None, :], sr, "approximate-spectral", False, None, [], warns)
