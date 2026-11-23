"""IR shaping with technically meaningful, linear, minimum-phase corrective filters.
Nothing here emulates amp dynamics: no attack/release knobs on dynamics, no saturation inside an IR."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.signal import fftconvolve
from .reconstruct import min_phase_from_mag, smooth_octave, _fade_tail


@dataclass
class ShapeParams:
    low_depth_db: float = 0.0        # low shelf  ~ 60-200 Hz
    low_mid_db: float = 0.0          # bell ~ 250-500 Hz
    mid_db: float = 0.0              # bell ~ 500-2000 Hz
    presence_db: float = 0.0         # bell ~ 2-6 kHz
    transient_sharpness_db: float = 0.0  # high shelf > 3 kHz (min-phase => sharper onset)
    resonance: float = 1.0           # 1 = untouched, >1 exaggerates peaks/notches, <1 flattens (1/6-oct deviation)
    attack_preserve_ms: float = 5.0  # leading part never touched by tail fade
    fade_ms: float = 8.0             # tail fade-out
    length_ms: float = 100.0
    phase: str = "as_estimated"      # as_estimated | minimum | linear
    output_gain_db: float = 0.0


def _bell(f, fc, bw_oct, g):
    return g * np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) / bw_oct) ** 2)


def gain_curve_db(f, p: ShapeParams):
    g = np.zeros_like(f)
    g += p.low_depth_db / (1 + (np.maximum(f, 1) / 150.0) ** 2)
    g += _bell(f, 350, 0.5, p.low_mid_db) + _bell(f, 1000, 0.8, p.mid_db) + _bell(f, 3500, 0.7, p.presence_db)
    g += p.transient_sharpness_db / (1 + (3000.0 / np.maximum(f, 1)) ** 2)
    return g


def shape_ir(ir, sr, p: ShapeParams):
    """ir: (C, N). Returns shaped IR (C, M) of length p.length_ms."""
    out = []
    M = int(sr * p.length_ms / 1000)
    nfft = 1 << int(np.ceil(np.log2(max(len(ir[0]), M) * 4)))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    for h in ir:
        H = np.fft.rfft(h, nfft)
        gdb = gain_curve_db(f, p)
        if abs(p.resonance - 1) > 1e-6:
            mag = np.abs(H) ** 2
            sm = smooth_octave(f, mag, 6)
            gdb += (p.resonance - 1) * 10 * np.log10((mag + 1e-20) / (sm + 1e-20))
        if p.phase == "minimum":
            hs = min_phase_from_mag(np.abs(H) * 10 ** (gdb / 20), nfft)
        elif p.phase == "linear":
            z = np.fft.irfft(np.abs(H) * 10 ** (gdb / 20), nfft)
            hs = np.roll(z, M // 2)
        else:
            eq = min_phase_from_mag(10 ** (gdb / 20), nfft)[: nfft // 2]
            hs = fftconvolve(h, eq)
        hs = np.concatenate([hs, np.zeros(max(0, M - len(hs)))])[:M]
        keep = int(sr * p.attack_preserve_ms / 1000); fade = max(8, int(sr * p.fade_ms / 1000))
        w = np.ones(M); s = max(keep, M - fade)
        w[s:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, M - s)))
        out.append(hs * w * 10 ** (p.output_gain_db / 20))
    return np.stack(out)


CANDIDATES = {
    "Balanced": ShapeParams(),
    "Aggressive": ShapeParams(mid_db=2.0, presence_db=2.5, transient_sharpness_db=1.5, resonance=1.2, low_depth_db=-1.0),
    "Tight_Punchy": ShapeParams(low_depth_db=-3.0, low_mid_db=-1.5, transient_sharpness_db=2.0, length_ms=60, fade_ms=6),
    "Deep_Thick": ShapeParams(low_depth_db=3.0, low_mid_db=2.0, presence_db=-1.0, length_ms=150, resonance=1.1),
    "Bright_Articulate": ShapeParams(presence_db=3.0, transient_sharpness_db=2.5, low_mid_db=-1.0),
    "Max_Similarity": ShapeParams(length_ms=200, fade_ms=12, attack_preserve_ms=10),  # no tonal shaping at all
}


def build_candidates(res_ir, sr, base_len_ms=None):
    out = {}
    for name, p in CANDIDATES.items():
        q = ShapeParams(**asdict(p))
        if base_len_ms and name not in ("Tight_Punchy", "Max_Similarity"):
            q.length_ms = base_len_ms
        out[name] = (shape_ir(res_ir, sr, q), q)
    return out
