"""Headless usage:
 python -m nigr.cli paired --di di.wav --target amp.wav --out irs/ [--method freq|time] [--ms 100] [--auto-reg]
 python -m nigr.cli approx --ref song.wav [--ref ...] [--src myguitar_through_plugin.wav] --out irs/"""
import argparse, os, json
from . import pipeline, reconstruct, optimize, export, analysis
from .io_utils import load_wav


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nigr"); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("paired"); a.add_argument("--di", action="append", required=True); a.add_argument("--target", action="append", required=True)
    a.add_argument("--method", default="freq"); a.add_argument("--ms", type=float, default=100); a.add_argument("--auto-reg", action="store_true")
    a.add_argument("--reg", type=float, default=1e-5); a.add_argument("--out", required=True)
    b = sp.add_parser("approx"); b.add_argument("--ref", action="append", required=True); b.add_argument("--src", action="append")
    b.add_argument("--ms", type=float, default=100); b.add_argument("--out", required=True)
    n = ap.parse_args(argv)
    sr = pipeline.SR; taps = int(sr * n.ms / 1000)
    if n.cmd == "paired":
        pairs = [pipeline.load_pair(d, t) for d, t in zip(n.di, n.target)]
        res = reconstruct.reconstruct_paired(pairs, sr, taps, n.method, n.reg, auto_reg=n.auto_reg)
        x, y = pairs[0]; r = pipeline.render(x, res.ir)
        print(json.dumps(pipeline.compare_channels(y, r), indent=1))
    else:
        refs = [load_wav(p, sr)[0].mean(0) for p in n.ref]; src = [load_wav(p, sr)[0].mean(0) for p in n.src] if n.src else None
        res = reconstruct.reconstruct_approximate(refs, sr, taps, src)
    for w in res.warnings: print("WARNING:", w)
    cands = optimize.build_candidates(res.ir, sr, n.ms)
    log = export.export_many(n.out, {k: v[0] for k, v in cands.items()}, sr, None)
    print("Exported:", ", ".join(cands), "->", n.out)

if __name__ == "__main__":
    main()
