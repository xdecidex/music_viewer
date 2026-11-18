"""System-audio capture (WASAPI loopback) and spectrum analysis.

A background thread records whatever is playing on the default output device
into a ring buffer. The render loop calls `analyze()` once per frame to get a
smoothed, 0..1-normalised spectrum (same scaling as the Web Audio AnalyserNode:
Blackman window, -100..-30 dB mapped to 0..1).
"""
import ctypes
import threading
import time
import warnings

import numpy as np
import soundcard as sc

SAMPLE_RATE = 48000
FFT_SIZE = 2048
BANDS = 64            # log-spaced bands for the bar ring
MIN_DB, MAX_DB = -100.0, -30.0
SMOOTHING = 0.72      # per-frame at 60 fps, like AnalyserNode.smoothingTimeConstant
BLOCK = 512


class AudioAnalyzer:
    def __init__(self):
        self.device_name = "starting..."
        self._buf = np.zeros(FFT_SIZE * 2, np.float32)
        self._lock = threading.Lock()
        self._last_update = 0.0
        self._running = True

        self._window = np.blackman(FFT_SIZE).astype(np.float32)
        self._mag = np.zeros(FFT_SIZE // 2, np.float32)
        self.spectrum = np.zeros(FFT_SIZE // 2, np.float32)

        hz_per_bin = SAMPLE_RATE / FFT_SIZE
        lo, hi = 30 / hz_per_bin, min(14000 / hz_per_bin, FFT_SIZE // 2 - 1)
        edges = lo * (hi / lo) ** (np.arange(BANDS + 1) / BANDS)
        self._band_ranges = []
        for i in range(BANDS):
            a = int(edges[i])
            b = max(a + 1, int(edges[i + 1]))
            self._band_ranges.append((a, b))

        def hz(f):
            return int(round(f / hz_per_bin))
        self._bass = (max(1, hz(25)), hz(180))
        self._mid = (hz(180), hz(2000))
        self._treble = (hz(2000), hz(9000))

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ capture thread
    def _run(self):
        warnings.simplefilter("ignore")  # soundcard warns on every data discontinuity
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COM for this thread
        except Exception:
            pass
        while self._running:
            try:
                speaker = sc.default_speaker()
                mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                self.device_name = speaker.name
                next_check = time.perf_counter() + 2.0
                with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK) as rec:
                    while self._running:
                        data = rec.record(numframes=BLOCK)
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        n = len(mono)
                        if n:
                            with self._lock:
                                self._buf[:-n] = self._buf[n:]
                                self._buf[-n:] = mono[-len(self._buf):]
                                self._last_update = time.perf_counter()
                        # follow the user switching their default output device
                        if time.perf_counter() > next_check:
                            next_check = time.perf_counter() + 2.0
                            if sc.default_speaker().name != speaker.name:
                                break
            except Exception as e:  # device unplugged, exclusive mode, etc.
                self.device_name = f"audio error: {e}"
                time.sleep(1.0)

    def close(self):
        self._running = False

    # ------------------------------------------------------------ analysis
    def analyze(self, dt):
        """Update and return the smoothed 0..1 spectrum (FFT_SIZE/2 bins)."""
        with self._lock:
            x = self._buf[-FFT_SIZE:].copy()
            stale = time.perf_counter() - self._last_update > 0.2
        if stale:  # WASAPI loopback delivers nothing while the system is silent
            x[:] = 0.0
        mag = np.abs(np.fft.rfft(x * self._window))[: FFT_SIZE // 2] / FFT_SIZE
        tau = SMOOTHING ** (dt * 60.0)
        self._mag = tau * self._mag + (1.0 - tau) * mag
        db = 20.0 * np.log10(self._mag + 1e-12)
        self.spectrum = np.clip((db - MIN_DB) / (MAX_DB - MIN_DB), 0.0, 1.0)
        return self.spectrum

    def bands(self):
        s = self.spectrum
        return np.array([s[a:b].mean() for a, b in self._band_ranges], np.float32)

    def levels(self):
        s = self.spectrum
        return (s[slice(*self._bass)].mean(),
                s[slice(*self._mid)].mean(),
                s[slice(*self._treble)].mean())
