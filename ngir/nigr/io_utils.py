"""WAV import/export. Uses `soundfile` when installed, otherwise a built-in RIFF reader/writer.
Audio arrays are always float64, shape (channels, samples). Source files are only ever read."""
from __future__ import annotations
import struct
from math import gcd
import numpy as np
from scipy.signal import resample_poly

try:
    import soundfile as _sf
except Exception:  # pragma: no cover
    _sf = None


def _read_riff(path):
    with open(path, "rb") as f:
        b = f.read()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("Incomplete WAV file")
    tag, ch, sr, bits = fmt
    if tag == 1 and bits == 16:
        a = np.frombuffer(data, "<i2").astype(np.float64) / 32768.0
    elif tag == 1 and bits == 24:
        n = len(data) // 3
        r = np.frombuffer(data[:n * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        v = r[:, 0] | (r[:, 1] << 8) | (r[:, 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        a = v.astype(np.float64) / 8388608.0
    elif tag == 1 and bits == 32:
        a = np.frombuffer(data, "<i4").astype(np.float64) / 2147483648.0
    elif tag == 3 and bits == 32:
        a = np.frombuffer(data, "<f4").astype(np.float64)
    elif tag == 3 and bits == 64:
        a = np.frombuffer(data, "<f8").astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV format tag={tag} bits={bits}")
    a = a[: (len(a) // ch) * ch].reshape(-1, ch).T
    return a, sr


def _write_pcm24(path, x, sr):
    x = np.atleast_2d(x)
    v = np.round(np.clip(x, -1.0, 8388607 / 8388608) * 8388608).astype(np.int32).T.reshape(-1)
    raw = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes()
    ch = x.shape[0]
    hdr = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 3, ch * 3, 24)
    hdr += b"data" + struct.pack("<I", len(raw))
    with open(path, "wb") as f:
        f.write(hdr + raw)


def load_wav(path, target_sr: int | None = None):
    """Returns (x[channels, n], sr). Optionally resamples to target_sr."""
    if _sf is not None:
        d, sr = _sf.read(path, dtype="float64", always_2d=True)
        x = d.T
    else:
        x, sr = _read_riff(path)
    if target_sr and sr != target_sr:
        x, sr = resample(x, sr, target_sr), target_sr
    return np.ascontiguousarray(x), sr


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1)


def save_wav(path, x, sr, subtype="PCM_24"):
    x = np.atleast_2d(x)
    if _sf is not None:
        _sf.write(path, x.T, sr, subtype=subtype)
    elif subtype == "PCM_24":
        _write_pcm24(path, x, sr)
    else:
        raise RuntimeError("Install `soundfile` for formats other than PCM_24")
