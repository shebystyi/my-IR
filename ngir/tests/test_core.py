import os, tempfile, unittest
import numpy as np
from scipy.signal import fftconvolve, lfilter
from nigr import io_utils, reconstruct, optimize, export, metrics, analysis, nonlinear, pipeline

SR = 48000


def guitar_di(seconds=12, seed=0):
    """Synthetic 'DI': plucked Karplus-Strong notes of varying pitch/level + very low noise floor."""
    rng = np.random.default_rng(seed); out = np.zeros(int(seconds * SR))
    for k in range(int(seconds * 3)):
        f0 = rng.choice([82.4, 110, 146.8, 196, 246.9, 329.6]); n = int(SR / f0)
        buf = rng.uniform(-1, 1, n) * rng.uniform(0.2, 1.0); y = np.zeros(int(0.9 * SR))
        for i in range(len(y)):
            y[i] = buf[i % n]; buf[i % n] = 0.5 * (buf[i % n] + buf[(i + 1) % n]) * 0.996
        s = int(k * SR / 3); seg = out[s:s + len(y)]; seg += y[:len(seg)]
    out += 1e-3 * rng.standard_normal(len(out))
    return out / np.abs(out).max() * 0.5


def known_ir(n=2400):
    t = np.arange(n) / SR; rng = np.random.default_rng(1)
    h = rng.standard_normal(n) * np.exp(-t / 0.006)
    h = lfilter([0.3], [1, -0.7], h)                     # darker
    h[0:5] += np.array([0.2, 0.5, 1.0, 0.4, 0.1]) * 3
    return h / np.abs(h).max()


class TestIO(unittest.TestCase):
    def test_roundtrip_24bit_and_resample(self):
        x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)[None]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.wav"); io_utils.save_wav(p, x, SR)
            y, sr = io_utils.load_wav(p); self.assertEqual(sr, SR)
            self.assertLess(np.max(np.abs(x - y)), 2e-7)
            y2, sr2 = io_utils.load_wav(p, 44100); self.assertEqual(sr2, 44100); self.assertAlmostEqual(len(y2[0]), 44100, delta=2)
            # resample accuracy: tone preserved
            ref = 0.5 * np.sin(2 * np.pi * 440 * np.arange(len(y2[0])) / 44100)
            self.assertLess(np.sqrt(np.mean((y2[0][2000:-2000] - ref[2000:-2000]) ** 2)), 1e-3)

    def test_stereo_roundtrip(self):
        x = np.random.default_rng(0).uniform(-.5, .5, (2, 1000))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.wav"); io_utils.save_wav(p, x, SR); y, _ = io_utils.load_wav(p)
            self.assertEqual(y.shape, (2, 1000)); self.assertLess(np.max(np.abs(x - y)), 2e-7)


class TestReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = guitar_di(); cls.h = known_ir(); cls.y = fftconvolve(cls.x, cls.h)[:len(cls.x)]

    def _check(self, method):
        res = reconstruct.reconstruct_paired([(self.x, self.y[None])], SR, 4800, method, 1e-7)
        h = res.ir[0]; start = 64 - res.delay_samples[0]   # estimator aligns the dominant peak at index 64
        est = h[start:start + len(self.h)]
        err = 20 * np.log10(np.linalg.norm(est - self.h[:len(est)]) / np.linalg.norm(self.h))
        self.assertLess(err, -35, f"{method}: IR error {err:.1f} dB")
        self.assertTrue(res.physical)
        self.assertGreater(res.coherence[1][0][(res.coherence[0] > 100) & (res.coherence[0] < 3000)].mean(), 0.95)

    def test_freq_recovers_known_fir(self): self._check("freq")
    def test_time_recovers_known_fir(self): self._check("time")

    def test_convolution_accuracy(self):
        a = np.random.default_rng(2).standard_normal(3000); b = np.random.default_rng(3).standard_normal(500)
        self.assertLess(np.max(np.abs(fftconvolve(a, b) - np.convolve(a, b))), 1e-9)

    def test_phase_preserved(self):
        # non-minimum-phase IR (peak late): estimator must keep the delay/phase
        h = np.zeros(1000); h[200] = 1.0; h[260] = 0.5; h[300] = -0.3
        y = fftconvolve(self.x, h)[:len(self.x)]
        r = reconstruct.reconstruct_paired([(self.x, y[None])], SR, 2400, "freq", 1e-8).ir[0]
        k = int(np.argmax(np.abs(r)))
        self.assertAlmostEqual(r[k + 60] / r[k], 0.5, delta=0.03); self.assertAlmostEqual(r[k + 100] / r[k], -0.3, delta=0.03)

    def test_noise_robust_and_stable(self):
        yn = self.y + 1e-3 * np.random.default_rng(5).standard_normal(len(self.y))
        res = reconstruct.reconstruct_paired([(self.x, yn[None])], SR, 4800, "freq", 1e-5)
        self.assertTrue(np.all(np.isfinite(res.ir))); self.assertLess(np.abs(res.ir).max(), 10)
        st = 64 - res.delay_samples[0]; est = res.ir[0][st:st + len(self.h)]
        self.assertLess(20 * np.log10(np.linalg.norm(est - self.h[:len(est)]) / np.linalg.norm(self.h)), -20)

    def test_stereo_target(self):
        y2 = np.stack([self.y, fftconvolve(self.x, known_ir()[::-1] * 0.5)[:len(self.x)]])
        res = reconstruct.reconstruct_paired([(self.x, y2)], SR, 2400, "freq", 1e-6)
        self.assertEqual(res.ir.shape[0], 2)

    def test_approximate_mode_is_flagged(self):
        res = reconstruct.reconstruct_approximate([self.y], SR, 4800)
        self.assertFalse(res.physical); self.assertTrue(any("NOT a physically recovered" in w for w in res.warnings))
        self.assertTrue(np.all(np.isfinite(res.ir)))


class TestNonlinear(unittest.TestCase):
    def test_nonlinear_detected(self):
        x = guitar_di(10); h = known_ir()
        lin = fftconvolve(x, h)[:len(x)]
        nl = fftconvolve(np.tanh(8 * x), h)[:len(x)]
        a = nonlinear.analyze_nonlinear(x, lin, SR); b = nonlinear.analyze_nonlinear(x, nl, SR)
        self.assertLess(a["nonlinear_plus_noise_vs_output_db"], -30)
        self.assertGreater(b["nonlinear_plus_noise_vs_output_db"], a["nonlinear_plus_noise_vs_output_db"] + 12)
        self.assertLess(b["slope_db_per_db"], a["slope_db_per_db"] - 0.1)  # compression indicator

    def test_nonlinear_ir_is_flagged_not_perfect(self):
        x = guitar_di(10); nl = fftconvolve(np.tanh(8 * x), known_ir())[:len(x)]
        res = reconstruct.reconstruct_paired([(x, nl[None])], SR, 4800)
        self.assertTrue(res.warnings)  # coherence warning

    def test_mastered_no_di_case(self):
        y = fftconvolve(np.tanh(8 * guitar_di(8)), known_ir())[:8 * SR]
        res = reconstruct.reconstruct_approximate([y], SR)
        self.assertFalse(res.physical)


class TestOptimizeExport(unittest.TestCase):
    def test_untouched_candidate_is_transparent(self):
        h = known_ir()[None]
        p = optimize.ShapeParams(length_ms=50, fade_ms=2, attack_preserve_ms=40)
        s = optimize.shape_ir(h, SR, p); self.assertAlmostEqual(np.max(np.abs(s)), np.max(np.abs(h)), delta=1e-6)
        self.assertLess(np.linalg.norm(s[0][:2000] - h[0][:2000]) / np.linalg.norm(h[0][:2000]), 0.02)

    def test_eq_params_change_the_right_band(self):
        h = known_ir()[None]; s = optimize.shape_ir(h, SR, optimize.ShapeParams(presence_db=6, length_ms=50, attack_preserve_ms=50))
        f = np.fft.rfftfreq(8192, 1 / SR); d = 20 * np.log10(np.abs(np.fft.rfft(s[0], 8192)) / (np.abs(np.fft.rfft(h[0][:2400], 8192)) + 1e-12))
        self.assertGreater(d[(f > 3000) & (f < 4000)].mean(), 4); self.assertLess(abs(d[(f > 100) & (f < 300)].mean()), 1)

    def test_candidates(self):
        c = optimize.build_candidates(known_ir()[None], SR, 100)
        self.assertEqual(len(c), 6); [self.assertTrue(np.all(np.isfinite(v[0]))) for v in c.values()]

    def test_export_properties(self):
        ir = np.stack([known_ir(), known_ir() * .5])
        with tempfile.TemporaryDirectory() as d:
            for ms in (20, 100, 200):
                p = os.path.join(d, f"{ms}.wav"); export.export_ir(p, ir, SR, ms, peak_dbfs=-1)
                y, sr = io_utils.load_wav(p); self.assertEqual(sr, 48000); self.assertEqual(y.shape, (2, int(SR * ms / 1000)))
                self.assertAlmostEqual(20 * np.log10(np.abs(y).max()), -1, delta=0.01)
                self.assertAlmostEqual(np.abs(y[1]).max() / np.abs(y[0]).max(), 0.5, delta=0.05)  # stereo ratio kept
            self.assertEqual(os.path.getsize(os.path.join(d, "20.wav")), 44 + 960 * 2 * 3)  # 24-bit

    def test_no_silent_clipping(self):
        ir, w = export.normalize_ir(known_ir()[None], -1, +6); self.assertLessEqual(np.abs(ir).max(), 0.9999); self.assertTrue(w)

    def test_export_resamples_to_48k(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.wav"); export.export_ir(p, known_ir()[None, :2000], 44100, None)
            _, sr = io_utils.load_wav(p); self.assertEqual(sr, 48000)


class TestAnalysisMetrics(unittest.TestCase):
    def test_analysis_runs_and_tags(self):
        r = analysis.analyze(guitar_di(6)[None], SR); names = {m[0]: m for m in r["metrics"]}
        self.assertIn("Peak", names); self.assertEqual(names["Peak"][3], "measured")
        self.assertIn("Fundamental / harmonics", names) if "Fundamental" not in names else None
        self.assertTrue(all(m[3] in ("measured", "estimated") for m in r["metrics"]))

    def test_clipping_detected(self):
        x = np.clip(3 * guitar_di(4), -1, 1)[None]; r = {m[0]: m[1] for m in analysis.analyze(x, SR)["metrics"]}
        self.assertTrue(r["Clipping detected"])

    def test_metrics_identity_and_difference(self):
        x = guitar_di(6); m0 = metrics.compare(x, x, SR); self.assertLess(m0["log_spectral_distance_db"], 1e-6); self.assertLess(m0["time_domain_nmse_db"], -100)
        m1 = metrics.compare(x, lfilter([0.2], [1, -0.8], x), SR); self.assertGreater(m1["log_spectral_distance_db"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
