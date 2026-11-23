"""PySide6 + pyqtgraph desktop GUI. All DSP lives in the other modules; this file only wires widgets."""
from __future__ import annotations
import os, sys
import numpy as np
from PySide6 import QtCore, QtWidgets as W
import pyqtgraph as pg
from scipy.signal import spectrogram

from . import pipeline, reconstruct, optimize, export, analysis, nonlinear, metrics
from .io_utils import load_wav

SR = pipeline.SR
pg.setConfigOptions(antialias=True, background="#15171c", foreground="#b8bcc8")


def plot(title="", logx=False, xl="", yl=""):
    p = pg.PlotWidget(title=title); p.showGrid(x=True, y=True, alpha=.25); p.setLabel("bottom", xl); p.setLabel("left", yl)
    if logx: p.setLogMode(x=True)
    return p


def table(headers):
    t = W.QTableWidget(0, len(headers)); t.setHorizontalHeaderLabels(headers); t.horizontalHeader().setStretchLastSection(True)
    t.setEditTriggers(W.QAbstractItemView.NoEditTriggers); return t


def fill(t, rows):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            t.setItem(i, j, W.QTableWidgetItem(f"{v:.3f}" if isinstance(v, float) else str(v)))
    t.resizeColumnsToContents()


class Main(W.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Neural Guitar IR Reconstruction Studio"); self.resize(1400, 860)
        self.pairs, self.refs = [], []          # pairs: dict(name, di, target, x, y) ; refs: dict(name, x)
        self.result = None; self.cands = {}; self.current = None; self.analyses = {}
        tabs = W.QTabWidget(); self.setCentralWidget(tabs)
        for name, fn in [("1 Import", self.tab_import), ("2 Analysis", self.tab_analysis), ("3 Profile", self.tab_profile),
                         ("4 IR Reconstruction", self.tab_recon), ("5 Nonlinear", self.tab_nl), ("6 A/B", self.tab_ab), ("7 Export", self.tab_export)]:
            w = W.QWidget(); fn(w); tabs.addTab(w, name)
        self.status = self.statusBar(); self.prog = W.QProgressBar(); self.prog.setMaximumWidth(220); self.status.addPermanentWidget(self.prog)
        self.setStyleSheet("QWidget{background:#1b1e25;color:#d5d8e0;font-size:12px}QPushButton{background:#2b3140;padding:6px 12px;border:1px solid #3b4357;border-radius:3px}"
                           "QPushButton:hover{background:#38415a}QTabBar::tab{padding:8px 14px;background:#232733}QTabBar::tab:selected{background:#34405e}"
                           "QTableWidget{background:#15171c}QHeaderView::section{background:#232733;color:#b8bcc8}")

    # -------------------------------------------------------------- helpers
    def busy(self, v, msg=""):
        self.prog.setRange(0, 100); self.prog.setValue(v); self.status.showMessage(msg); QtWidgets_process()
    def warn(self, text): W.QMessageBox.warning(self, "Warning", text)

    # -------------------------------------------------------------- 1 import
    def tab_import(self, w):
        l = W.QVBoxLayout(w)
        l.addWidget(W.QLabel("Add <b>DI + processed pairs</b> (physical IR reconstruction possible) or <b>reference recordings without DI</b> "
                             "(approximate spectral match only). Source files are only read, never modified. WAV 16/24/32-bit PCM or 32-bit float, mono/stereo."))
        b = W.QHBoxLayout(); a1 = W.QPushButton("Add DI + processed pair…"); a2 = W.QPushButton("Add reference(s) without DI…"); r = W.QPushButton("Remove selected")
        a1.clicked.connect(self.add_pair); a2.clicked.connect(self.add_refs); r.clicked.connect(self.remove_sel)
        for x in (a1, a2, r): b.addWidget(x)
        b.addStretch(); l.addLayout(b); self.lst = W.QListWidget(); l.addWidget(self.lst)
        self.wave = plot("Waveform", xl="s"); l.addWidget(self.wave, 1); self.lst.currentRowChanged.connect(self.show_wave)

    def all_items(self): return [("PAIR", p) for p in self.pairs] + [("REF", r) for r in self.refs]

    def add_pair(self):
        di, _ = W.QFileDialog.getOpenFileName(self, "DI recording", "", "WAV (*.wav)")
        if not di: return
        t, _ = W.QFileDialog.getOpenFileName(self, "Processed (target) recording", "", "WAV (*.wav)")
        if not t: return
        try: x, y = pipeline.load_pair(di, t)
        except Exception as e: return self.warn(str(e))
        self.pairs.append(dict(name=f"{os.path.basename(di)} → {os.path.basename(t)}", x=x, y=y)); self.refresh_list()

    def add_refs(self):
        fs, _ = W.QFileDialog.getOpenFileNames(self, "Reference recordings", "", "WAV (*.wav)")
        for f in fs:
            try: x, _ = load_wav(f, SR)
            except Exception as e: self.warn(f"{f}: {e}"); continue
            self.refs.append(dict(name=os.path.basename(f), x=x))
        self.refresh_list()

    def remove_sel(self):
        i = self.lst.currentRow()
        if i < 0: return
        (self.pairs.pop(i) if i < len(self.pairs) else self.refs.pop(i - len(self.pairs))); self.refresh_list()

    def refresh_list(self):
        self.lst.clear(); [self.lst.addItem(("[PAIR] " if k == "PAIR" else "[REF]  ") + d["name"]) for k, d in self.all_items()]
        if hasattr(self, "cmb"): self.cmb.clear(); self.cmb.addItems([d["name"] for _, d in self.all_items()])

    def sig(self, i):
        k, d = self.all_items()[i]; return (d["y"] if k == "PAIR" else d["x"])

    def show_wave(self, i):
        if i < 0: return
        x = self.sig(i).mean(0); step = max(1, len(x) // 20000); self.wave.clear()
        t = np.arange(0, len(x), step) / SR; self.wave.plot(t, x[::step], pen="#5aa9ff")

    # -------------------------------------------------------------- 2 analysis
    def tab_analysis(self, w):
        l = W.QGridLayout(w); self.cmb = W.QComboBox(); btn = W.QPushButton("Analyze selected"); btn.clicked.connect(self.run_analysis)
        l.addWidget(self.cmb, 0, 0); l.addWidget(btn, 0, 1)
        self.tbl = table(["Property", "Value", "Unit", "Kind"]); l.addWidget(self.tbl, 1, 0, 3, 1)
        self.p_spec = plot("Spectrum (long-term)", True, "Hz", "dB"); self.p_env = plot("Dynamic envelope", xl="s", yl="dB"); self.p_sg = pg.PlotWidget(title="Spectrogram")
        l.addWidget(self.p_spec, 1, 1); l.addWidget(self.p_env, 2, 1); l.addWidget(self.p_sg, 3, 1); l.setColumnStretch(0, 1); l.setColumnStretch(1, 1)

    def run_analysis(self):
        i = self.cmb.currentIndex()
        if i < 0: return self.warn("Import something first.")
        x = self.sig(i); self.busy(20, "Analyzing…"); r = analysis.analyze(x, SR); self.analyses[self.cmb.currentText()] = r
        fill(self.tbl, r["metrics"]); f, p = r["ltas"]; self.p_spec.clear(); self.p_spec.plot(np.log10(f[1:]), p[1:], pen="#5aa9ff")
        e = r["envelope"]; s = max(1, len(e) // 10000); self.p_env.clear(); self.p_env.plot(np.arange(0, len(e), s) / SR, 20 * np.log10(e[::s] + 1e-9), pen="#ffb454")
        m = x.mean(0); ff, tt, S = spectrogram(m, SR, nperseg=2048, noverlap=1536); img = pg.ImageItem(10 * np.log10(S[:400] + 1e-12).T)
        self.p_sg.clear(); self.p_sg.addItem(img); img.setLevels((-120, -20)); img.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.busy(100, "Analysis done")

    # -------------------------------------------------------------- 3 profile
    def tab_profile(self, w):
        l = W.QVBoxLayout(w); b = W.QPushButton("Build combined reference profile from all analysed recordings"); b.clicked.connect(self.build_profile)
        l.addWidget(b); self.prof_plot = plot("Combined long-term spectrum", True, "Hz", "dB"); l.addWidget(self.prof_plot)
        self.prof_txt = W.QPlainTextEdit(); self.prof_txt.setReadOnly(True); l.addWidget(self.prof_txt)

    def build_profile(self):
        items = self.all_items(); res = []
        for i, (_, d) in enumerate(items):
            if d["name"] not in self.analyses: self.analyses[d["name"]] = analysis.analyze(self.sig(i), SR)
            res.append(self.analyses[d["name"]])
        if not res: return self.warn("Nothing imported.")
        f, P = analysis.combined_profile(res); self.prof_plot.clear(); self.prof_plot.plot(np.log10(f[1:]), P[1:], pen="#5aa9ff")
        lines = [f"Recordings analysed: {len(res)}", "MEASURED = directly computed; ESTIMATED = material-dependent / heuristic.", ""]
        for (_, d), r in zip(items, res):
            lines.append(f"== {d['name']}"); lines += [f"  [{k[:4]}] {n}: {v if not isinstance(v, float) else round(v, 2)} {u}" for n, v, u, k in r["metrics"]]
        self.prof_txt.setPlainText("\n".join(lines))

    # -------------------------------------------------------------- 4 reconstruction
    def tab_recon(self, w):
        l = W.QHBoxLayout(w); left = W.QVBoxLayout(); l.addLayout(left, 0); right = W.QVBoxLayout(); l.addLayout(right, 1)
        self.mode = W.QComboBox(); self.mode.addItems(["Paired – frequency-domain (Wiener/Tikhonov)", "Paired – time-domain (Toeplitz LS)", "Approximate – no DI (NOT a recovered IR)"])
        self.len_ms = W.QDoubleSpinBox(); self.len_ms.setRange(5, 400); self.len_ms.setValue(100); self.len_ms.setSuffix(" ms")
        self.reg = W.QDoubleSpinBox(); self.reg.setDecimals(2); self.reg.setRange(-12, 0); self.reg.setValue(-5); self.reg.setPrefix("10^")
        self.auto_reg = W.QCheckBox("Auto-select regularisation (held-out validation)")
        f = W.QFormLayout(); f.addRow("Mode", self.mode); f.addRow("IR length", self.len_ms); f.addRow("Regularisation λ", self.reg); f.addRow(self.auto_reg); left.addLayout(f)
        self.sh = {}
        g = W.QGroupBox("Shaping (linear, minimum-phase EQ on the IR – no dynamics)"); gf = W.QFormLayout(g)
        for key, lo, hi, v in [("low_depth_db", -9, 9, 0), ("low_mid_db", -9, 9, 0), ("mid_db", -9, 9, 0), ("presence_db", -9, 9, 0),
                               ("transient_sharpness_db", -9, 9, 0), ("resonance", 0, 2, 1), ("attack_preserve_ms", 0, 50, 5), ("output_gain_db", -24, 12, 0)]:
            s = W.QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(v); s.setSingleStep(.5); self.sh[key] = s; gf.addRow(key.replace("_", " "), s)
        self.phase = W.QComboBox(); self.phase.addItems(["as_estimated", "minimum", "linear"]); gf.addRow("phase", self.phase); left.addWidget(g)
        b1 = W.QPushButton("Reconstruct IR + build 6 candidates"); b1.clicked.connect(self.run_recon); b2 = W.QPushButton("Apply manual shaping to current IR"); b2.clicked.connect(self.apply_shape)
        left.addWidget(b1); left.addWidget(b2); self.cand_list = W.QListWidget(); self.cand_list.currentTextChanged.connect(self.pick_cand); left.addWidget(self.cand_list); left.addStretch()
        self.warn_lbl = W.QLabel(); self.warn_lbl.setWordWrap(True); self.warn_lbl.setStyleSheet("color:#ffb454"); right.addWidget(self.warn_lbl)
        self.p_ir = plot("IR waveform", xl="ms"); self.p_fr = plot("Frequency response", True, "Hz", "dB"); self.p_coh = plot("Coherence (1 = linear model fully valid)", True, "Hz")
        for p in (self.p_ir, self.p_fr, self.p_coh): right.addWidget(p)
        self.ir_txt = W.QLabel(); right.addWidget(self.ir_txt)

    def run_recon(self):
        mode = self.mode.currentIndex(); taps = int(SR * self.len_ms.value() / 1000)
        try:
            self.busy(10, "Reconstructing…")
            if mode < 2:
                if not self.pairs: return self.warn("Add at least one DI + processed pair.")
                self.result = reconstruct.reconstruct_paired([(p["x"], p["y"]) for p in self.pairs], SR, taps, "freq" if mode == 0 else "time", 10 ** self.reg.value(), auto_reg=self.auto_reg.isChecked())
            else:
                if not self.refs: return self.warn("Add reference recordings (no DI).")
                self.result = reconstruct.reconstruct_approximate([r["x"].mean(0) for r in self.refs], SR, taps)
            self.busy(70, "Building candidates…"); self.cands = optimize.build_candidates(self.result.ir, SR, self.len_ms.value())
        except Exception as e: return self.warn(f"{type(e).__name__}: {e}")
        self.cand_list.clear(); self.cand_list.addItems(list(self.cands)); self.cand_list.setCurrentRow(5)
        self.warn_lbl.setText(("⚠ " + "\n⚠ ".join(self.result.warnings)) if self.result.warnings else "Physical paired estimate. Coherence plot shows where the linear model is trustworthy.")
        if self.result.coherence:
            fr, c = self.result.coherence; self.p_coh.clear(); self.p_coh.plot(np.log10(fr[1:]), c[0][1:], pen="#6fdc8c"); self.p_coh.setYRange(0, 1)
        self.busy(100, "Done")

    def pick_cand(self, name):
        if name in self.cands: self.current = self.cands[name][0]; self.show_ir(self.current)

    def apply_shape(self):
        if self.result is None: return self.warn("Reconstruct first.")
        p = optimize.ShapeParams(**{k: s.value() for k, s in self.sh.items()}, length_ms=self.len_ms.value(), phase=self.phase.currentText())
        self.cands["Manual"] = (optimize.shape_ir(self.result.ir, SR, p), p)
        if self.cand_list.findItems("Manual", QtCore.Qt.MatchExactly) == []: self.cand_list.addItem("Manual")
        self.cand_list.setCurrentRow(self.cand_list.count() - 1)

    def show_ir(self, ir):
        h = ir[0]; t = np.arange(len(h)) / SR * 1000; self.p_ir.clear(); self.p_ir.plot(t, h, pen="#5aa9ff")
        n = 16384; H = 20 * np.log10(np.abs(np.fft.rfft(h, n)) + 1e-9); f = np.fft.rfftfreq(n, 1 / SR)
        self.p_fr.clear(); self.p_fr.plot(np.log10(f[1:]), H[1:], pen="#ffb454"); self.p_fr.setXRange(np.log10(30), np.log10(20000))
        self.ir_txt.setText("   |   ".join(f"{n}: {v:.2f} {u}" if isinstance(v, float) else f"{n}: {v}" for n, v, u, _ in analysis.ir_report(h, SR)[:4]))

    # -------------------------------------------------------------- 5 nonlinear
    def tab_nl(self, w):
        l = W.QVBoxLayout(w)
        l.addWidget(W.QLabel("Nonlinear analysis is <b>separate</b> from the IR: it never changes an exported IR. Needs a DI + processed pair. Results are indicators (estimated)."))
        b = W.QPushButton("Analyze nonlinearity of first pair"); b.clicked.connect(self.run_nl); e = W.QPushButton("Export time-aligned pair for NAM-style training…"); e.clicked.connect(self.export_nam)
        l.addWidget(b); l.addWidget(e); self.p_gain = plot("Gain vs input level (compression indicator)", xl="input dBFS (RMS 20 ms)", yl="gain dB"); l.addWidget(self.p_gain)
        self.nl_txt = W.QPlainTextEdit(); self.nl_txt.setReadOnly(True); l.addWidget(self.nl_txt)

    def run_nl(self):
        if not self.pairs: return self.warn("Needs a DI + processed pair.")
        p = self.pairs[0]; self.busy(30, "Nonlinear analysis…"); r = nonlinear.analyze_nonlinear(p["x"], p["y"][0], SR); self.nl_r = r
        if "gain_vs_input_level" in r:
            c = np.array(r["gain_vs_input_level"]); self.p_gain.clear(); self.p_gain.plot(c[:, 0], c[:, 1], pen="#ff6b6b", symbol="o")
        self.nl_txt.setPlainText("\n".join(f"{k}: {v}" for k, v in r.items() if k != "gain_vs_input_level")); self.busy(100, "Done")

    def export_nam(self):
        if not self.pairs: return
        d = W.QFileDialog.getExistingDirectory(self, "Output folder")
        if d: nonlinear.export_nam_pair(d, self.pairs[0]["x"], self.pairs[0]["y"][0], SR); self.status.showMessage("Aligned pair written.")

    # -------------------------------------------------------------- 6 A/B
    def tab_ab(self, w):
        l = W.QVBoxLayout(w); h = W.QHBoxLayout(); self.ab_c = W.QComboBox(); b = W.QPushButton("Compute comparison (first pair)"); b.clicked.connect(self.run_ab)
        pr = W.QPushButton("▶ Reference"); pi = W.QPushButton("▶ DI through IR (loudness-matched)"); st = W.QPushButton("■ Stop")
        pr.clicked.connect(lambda: self.play("ref")); pi.clicked.connect(lambda: self.play("ir")); st.clicked.connect(self.stop)
        for x in (self.ab_c, b, pr, pi, st): h.addWidget(x)
        l.addLayout(h); self.ab_tbl = table(["Metric", "Value"]); self.p_ab = plot("Spectrum: reference vs IR output", True, "Hz", "dB"); l.addWidget(self.ab_tbl); l.addWidget(self.p_ab)

    def run_ab(self):
        if not self.pairs or not self.cands: return self.warn("Needs a pair and a reconstructed IR.")
        name = self.cand_list.currentItem().text() if self.cand_list.currentItem() else "Max_Similarity"; ir = self.cands[name][0]
        p = self.pairs[0]; r = pipeline.render(p["x"], ir); tgt, est = p["y"][0], r[0]
        from .reconstruct import align_pair
        est, tgt, _ = align_pair(est, tgt, 0); m = metrics.compare(tgt, est, SR); est_m, _ = metrics.match_loudness(tgt, est)
        self.ab_ref, self.ab_irs = tgt, est_m
        rows = [(k, v) for k, v in m.items() if k != "band_error_db"] + [(f"Band error {k} (IR − ref)", f"{v:+.2f} dB") for k, v in m["band_error_db"].items()]
        rows.append(("Reproduced vs different", "Bands within ±1.5 dB: " + ", ".join(k for k, v in m["band_error_db"].items() if abs(v) <= 1.5) + " | different: " + (", ".join(k for k, v in m["band_error_db"].items() if abs(v) > 1.5) or "none")))
        fill(self.ab_tbl, rows); f, a = metrics.ltas(tgt, SR); _, b = metrics.ltas(est_m, SR); self.p_ab.clear(); self.p_ab.addLegend()
        self.p_ab.plot(np.log10(f[1:]), 10 * np.log10(a[1:] + 1e-20), pen="#5aa9ff", name="reference"); self.p_ab.plot(np.log10(f[1:]), 10 * np.log10(b[1:] + 1e-20), pen="#ffb454", name="DI × IR")

    def play(self, which):
        try: import sounddevice as sd
        except Exception: return self.warn("Install `sounddevice` for playback (pip install sounddevice).")
        if not hasattr(self, "ab_ref"): return self.warn("Compute comparison first.")
        sd.play((self.ab_ref if which == "ref" else self.ab_irs).astype(np.float32) * 0.5, SR)

    def stop(self):
        try: import sounddevice as sd; sd.stop()
        except Exception: pass

    # -------------------------------------------------------------- 7 export
    def tab_export(self, w):
        f = W.QFormLayout(w); self.ex_dur = W.QComboBox(); self.ex_dur.addItems(list(export.DURATION_PRESETS_MS) + ["Full length"]); self.ex_dur.setCurrentIndex(1)
        self.ex_peak = W.QDoubleSpinBox(); self.ex_peak.setRange(-24, 0); self.ex_peak.setValue(-1); self.ex_peak.setSuffix(" dBFS peak")
        self.ex_gain = W.QDoubleSpinBox(); self.ex_gain.setRange(-24, 12); self.ex_gain.setSuffix(" dB extra gain")
        self.ex_all = W.QCheckBox("Export all candidates"); self.ex_all.setChecked(True); self.ex_mono = W.QCheckBox("Force mono (sum channels)")
        b = W.QPushButton("Export…"); b.clicked.connect(self.do_export); self.ex_log = W.QPlainTextEdit(); self.ex_log.setReadOnly(True)
        f.addRow("Duration", self.ex_dur); f.addRow("Normalisation", self.ex_peak); f.addRow("Gain", self.ex_gain); f.addRow(self.ex_all); f.addRow(self.ex_mono); f.addRow(b); f.addRow(self.ex_log)
        f.addRow(W.QLabel("48 kHz / 24-bit PCM WAV. No limiter is applied: if the gain would exceed 0 dBFS it is reduced and you are told.\n"
                          "Always verify level/length in your Neural DSP plugin's IR loader."))

    def do_export(self):
        if not self.cands: return self.warn("Reconstruct an IR first.")
        d = W.QFileDialog.getExistingDirectory(self, "Export folder")
        if not d: return
        t = self.ex_dur.currentText(); ms = export.DURATION_PRESETS_MS.get(t)
        sel = self.cands if self.ex_all.isChecked() else {self.cand_list.currentItem().text(): self.cands[self.cand_list.currentItem().text()]}
        log = export.export_many(d, {k: v[0] for k, v in sel.items()}, SR, ms, peak_dbfs=self.ex_peak.value(), extra_gain_db=self.ex_gain.value(), mono_only=self.ex_mono.isChecked())
        label = "PHYSICAL (paired)" if self.result.physical else "APPROXIMATE – not a recovered IR"
        self.ex_log.setPlainText(f"Exported {len(log)} file(s) to {d}  [{label}]\n" + "\n".join(f"{k}: {'; '.join(v) or 'ok'}" for k, v in log.items()))


def QtWidgets_process(): W.QApplication.processEvents()


def main():
    app = W.QApplication(sys.argv); m = Main(); m.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
