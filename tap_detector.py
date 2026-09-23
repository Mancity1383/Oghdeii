import math
import threading
import time
from collections import Counter

import numpy as np

try:
    from pynput import keyboard as _pynput_keyboard
except Exception:
    _pynput_keyboard = None


class DetectedTapGesture:
    """A finalized 1/2/3-tap gesture."""

    def __init__(self, count, side="center", peak=0.0):
        self.count = int(count)
        self.side = side if side in ("left", "right", "center") else "center"
        self.peak = float(peak)
        # Wall-clock timestamp is useful to callers/logs. Internal timing uses monotonic().
        self.timestamp = time.time()

    def name(self):
        side_prefix = f"{self.side}_" if self.side in ("left", "right") else ""
        if self.count == 1:
            return f"{side_prefix}single_tap"
        if self.count == 2:
            return f"{side_prefix}double_tap"
        if self.count == 3:
            return f"{side_prefix}triple_tap"
        return f"{side_prefix}{self.count}_tap"

    def __repr__(self):
        return (
            f"<TapGesture {self.name()} count={self.count} "
            f"side={self.side} peak={self.peak:.3f}>"
        )


class TapDetector:
    """
    Real-time chassis-tap detector.

    The detector combines an adaptive amplitude gate with low/high-band energy,
    impulsiveness (crest factor), attack rise time, a speech-periodicity guard,
    refractory debouncing, optional keyboard/speech suppression, and a
    race-safe multi-tap state machine.

    Important: thresholds are intentionally configurable because microphone
    gain, chassis construction, DSP/driver processing, and microphone placement
    vary substantially between laptops.
    """

    def __init__(
        self,
        config_manager,
        on_gesture_detected=None,
        on_level_update=None,
        sample_rate=48000,
        monitor_keyboard=True,
        on_candidate_evaluated=None,
    ):
        self.config_manager = config_manager
        self.on_gesture_detected = on_gesture_detected
        self.on_level_update = on_level_update
        self.on_candidate_evaluated = on_candidate_evaluated
        self.rejection_counts = Counter()
        self.last_candidate = None

        # Level telemetry is display-only; throttle to ~60 Hz instead of one
        # cross-thread Qt signal per audio block (~187/s at 48 kHz/256).
        self._last_level_emit = 0.0

        self.sample_rate = self._validated_sample_rate(sample_rate)
        self._history_seconds = 0.060  # enough history for attack + pitch analysis
        self._allocate_buffers()

        # Adaptive low-band peak floor. It is updated only from non-event
        # background (or extremely slowly during long stationary noise).
        self.noise_floor = 0.0015
        self.noise_alpha_down = 0.020
        self.noise_alpha_up = 0.004

        # Gesture state.
        self.last_tap_time = float("-inf")
        self.current_taps = 0
        self.current_side = "center"
        self.max_peak_in_group = 0.0
        self._side_log_sum = 0.0
        self._side_observations = 0

        self.group_timer = None
        self.reset_timer = None
        self._timer_generation = 0
        self.lock = threading.RLock()

        # Acoustic duration / suppression state.
        self._loud_duration_s = 0.0
        self.is_in_sustained_sound = False
        self.sustained_suppression_until = 0.0
        self.voice_like_suppression_until = 0.0
        self.post_gesture_suppression_until = 0.0
        self.voice_speech_active_until = 0.0

        # Keyboard suppression.
        self.last_key_time = float("-inf")
        self.keyboard_listener = None
        self.monitor_keyboard = bool(monitor_keyboard)
        self._stopped = False

    # ------------------------------------------------------------------
    # Configuration / lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_sample_rate(value):
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            value = 48000
        return max(8000, value)

    def _cfg_float(self, key, default, low=None, high=None):
        try:
            value = float(self.config_manager.get(key, default))
        except Exception:
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        return value

    def _cfg_bool(self, key, default):
        try:
            return bool(self.config_manager.get(key, default))
        except Exception:
            return bool(default)

    @property
    def sensitivity(self):
        return self._cfg_float("sensitivity", 0.90, 0.10, 1.0)

    @property
    def grouping_window_s(self):
        return self._cfg_float("grouping_window_ms", 400.0, 80.0, 1500.0) / 1000.0

    @property
    def refractory_period_s(self):
        return self._cfg_float("refractory_period_ms", 110.0, 20.0, 1000.0) / 1000.0

    @property
    def suppress_typing(self):
        return self._cfg_bool("suppress_typing", True)

    @property
    def typing_cooldown_s(self):
        return self._cfg_float("typing_cooldown_ms", 400.0, 0.0, 3000.0) / 1000.0

    @property
    def detect_sides(self):
        return self._cfg_bool("detect_sides", True)

    @property
    def crowded_mode(self):
        return self._cfg_bool("crowded_mode", False)

    @property
    def max_hf_ratio(self):
        return self._cfg_float("max_hf_ratio", 0.36, 0.0, 20.0)

    @property
    def max_rise_time_ms(self):
        return self._cfg_float("max_rise_time_ms", 4.5, 0.2, 30.0)

    @property
    def min_crest_factor(self):
        return self._cfg_float("min_crest_factor", 2.05, 1.05, 20.0)

    @property
    def max_harmonicity(self):
        # A slightly more conservative default than the old 0.38 reduces
        # rejection of resonant metal/plastic chassis taps.
        return self._cfg_float("max_harmonicity", 0.45, 0.05, 0.99)

    @property
    def side_ratio(self):
        return self._cfg_float("side_ratio", 1.35, 1.01, 5.0)

    @property
    def max_sustained_s(self):
        return self._cfg_float("max_sustained_ms", 55.0, 15.0, 500.0) / 1000.0

    @property
    def sustained_recovery_s(self):
        return self._cfg_float("sustained_recovery_ms", 120.0, 50.0, 2000.0) / 1000.0

    @property
    def sustained_revoke_s(self):
        return self._cfg_float("sustained_revoke_ms", 140.0, 60.0, 1000.0) / 1000.0

    @property
    def post_gesture_cooldown_s(self):
        return self._cfg_float("post_gesture_cooldown_ms", 180.0, 0.0, 1000.0) / 1000.0

    @property
    def triple_confirm_s(self):
        return self._cfg_float("triple_confirm_ms", 220.0, 140.0, 1000.0) / 1000.0

    @property
    def voice_like_hold_s(self):
        return self._cfg_float("voice_like_hold_ms", 900.0, 100.0, 5000.0) / 1000.0

    def _allocate_buffers(self):
        # Keep the history duration stable when the device sample rate changes.
        self.buf_len = max(512, int(round(self.sample_rate * self._history_seconds)))
        self.buf_low_l = np.zeros(self.buf_len, dtype=np.float32)
        self.buf_low_r = np.zeros(self.buf_len, dtype=np.float32)
        self.buf_high_l = np.zeros(self.buf_len, dtype=np.float32)
        self.buf_high_r = np.zeros(self.buf_len, dtype=np.float32)

    def set_sample_rate(self, sample_rate):
        """
        Synchronize timing-dependent analysis with the actual audio device rate.

        AudioEngine can choose the device's native sample rate at start-up, so
        callers should invoke this after the stream has been opened (or cheaply
        before each process_chunk; unchanged values are ignored).
        """
        new_rate = self._validated_sample_rate(sample_rate)
        if new_rate == self.sample_rate:
            return False

        with self.lock:
            self.sample_rate = new_rate
            self._allocate_buffers()
            self._loud_duration_s = 0.0
            self.is_in_sustained_sound = False
            self._invalidate_timers_locked()
            self._clear_group_locked()
        return True

    def _start_keyboard_listener(self):
        if not self.monitor_keyboard or _pynput_keyboard is None or self.keyboard_listener is not None:
            return

        try:
            def on_press(_key):
                now = time.monotonic()
                self.last_key_time = now
                # A keyboard impact can reach the microphone a few ms before
                # pynput delivers the event. Cancel an already-pending gesture
                # as well as suppressing subsequent audio blocks.
                if self.suppress_typing:
                    self._discard_pending_group()

            self.keyboard_listener = _pynput_keyboard.Listener(on_press=on_press)
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
        except Exception as exc:
            self.keyboard_listener = None
            print(f"[TapDetector] Could not start keyboard listener: {exc}")

    def start(self):
        """Enable detector lifecycle resources for knock mode."""
        with self.lock:
            self._stopped = False
        self._start_keyboard_listener()
        return True

    def pause(self):
        """Release the global keyboard hook while keeping the detector reusable."""
        with self.lock:
            self._invalidate_timers_locked()
            self._clear_group_locked()
        listener, self.keyboard_listener = self.keyboard_listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

    def stop(self):
        """Release the keyboard hook and invalidate all pending timer callbacks."""
        with self.lock:
            if self._stopped:
                return
            self._stopped = True
            self._invalidate_timers_locked()
            self._clear_group_locked()

        listener, self.keyboard_listener = self.keyboard_listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

    close = stop

    def reset(self):
        """Reset transient detector state without destroying callbacks/config."""
        with self.lock:
            self._invalidate_timers_locked()
            self._clear_group_locked()
            self.last_tap_time = float("-inf")
            self._loud_duration_s = 0.0
            self.is_in_sustained_sound = False
            self.sustained_suppression_until = 0.0
            self.voice_like_suppression_until = 0.0
            self.post_gesture_suppression_until = 0.0
            self.voice_speech_active_until = 0.0
            self.buf_low_l.fill(0.0)
            self.buf_low_r.fill(0.0)
            self.buf_high_l.fill(0.0)
            self.buf_high_r.fill(0.0)

    def reset_diagnostics(self):
        with self.lock:
            self.rejection_counts.clear()
            self.last_candidate = None

    def get_diagnostics(self):
        with self.lock:
            return {
                "counts": dict(self.rejection_counts),
                "last_candidate": dict(self.last_candidate) if self.last_candidate else None,
                "noise_floor": float(self.noise_floor),
                "threshold": float(self.calculate_threshold()),
            }

    def _record_candidate(self, result, **metrics):
        record = {"result": result, "timestamp": time.time(), **metrics}
        with self.lock:
            self.rejection_counts[result] += 1
            self.last_candidate = record
        callback = self.on_candidate_evaluated
        if callback:
            try:
                callback(record)
            except Exception:
                pass

    def notify_speech_active(self, duration_s=0.60):
        """
        Suppress knock detection while speech is active.

        Pending taps are discarded too; otherwise a speech transient registered
        just before the recognizer reports speech could still dispatch later.
        """
        try:
            duration_s = max(0.0, float(duration_s))
        except (TypeError, ValueError):
            duration_s = 0.60

        now = time.monotonic()
        with self.lock:
            self.voice_speech_active_until = max(
                self.voice_speech_active_until, now + duration_s
            )
            self._invalidate_timers_locked()
            self._clear_group_locked()

    # ------------------------------------------------------------------
    # Signal features
    # ------------------------------------------------------------------

    def calculate_threshold(self):
        """
        Adaptive low-band peak threshold.

        Sensitivity 0.10 maps to the least-sensitive absolute threshold and
        sensitivity 1.00 to the most-sensitive threshold. A noise-floor
        multiplier prevents the threshold from collapsing into background noise.
        """
        sens = self.sensitivity
        normalized = (sens - 0.10) / 0.90
        min_thresh = 0.008
        max_thresh = 0.120
        base = max_thresh - normalized * (max_thresh - min_thresh)

        # At high sensitivity the dynamic multiplier is slightly lower.
        noise_multiplier = 3.4 - 0.4 * normalized
        return max(base, self.noise_floor * noise_multiplier, min_thresh)

    def _update_noise_floor(self, current_peak, threshold, sustained=False):
        if not math.isfinite(current_peak) or current_peak < 0.0:
            return

        if current_peak < threshold * 0.85:
            alpha = (
                self.noise_alpha_down
                if current_peak < self.noise_floor
                else self.noise_alpha_up
            )
            self.noise_floor += alpha * (current_peak - self.noise_floor)
        elif sustained and self._loud_duration_s > 0.25:
            # Very slow upward adaptation for stationary background noise.
            # Clamp the sample so speech/music cannot instantly desensitize the
            # detector.
            capped = min(current_peak, threshold * 1.25)
            if capped > self.noise_floor:
                self.noise_floor += 0.0008 * (capped - self.noise_floor)

        self.noise_floor = float(max(1e-6, min(self.noise_floor, 1.0)))

    def _compute_harmonicity(self, signal):
        """
        Return normalized autocorrelation strength in the 80-400 Hz lag range.

        FFT autocorrelation avoids the O(N^2) np.correlate cost in the audio
        callback path.
        """
        x = np.asarray(signal, dtype=np.float64).reshape(-1)
        n = x.size
        if n < 64:
            return 0.0

        min_lag = max(1, int(self.sample_rate / 400.0))
        max_lag = min(n - 2, int(self.sample_rate / 80.0))
        if min_lag >= max_lag:
            return 0.0

        x = x - np.mean(x)
        energy = float(np.dot(x, x))
        if energy < 1e-10:
            return 0.0

        nfft = 1 << ((2 * n - 1).bit_length())
        spectrum = np.fft.rfft(x, n=nfft)
        corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:n]
        values = corr[min_lag : max_lag + 1]
        if values.size == 0:
            return 0.0

        # corr[0] is the signal energy. Clipping protects against tiny floating
        # point overshoots above 1.
        return float(np.clip(np.max(values) / (corr[0] + 1e-12), 0.0, 1.0))

    def _compute_rise_time_ms(self, signal, peak_idx, peak_val):
        """
        Estimate 20%-to-peak attack time using the contiguous rise before peak.
        """
        if peak_idx <= 0 or peak_val < 1e-7:
            return 0.0

        envelope = np.abs(np.asarray(signal).reshape(-1))
        peak_idx = min(int(peak_idx), envelope.size - 1)
        threshold_20 = 0.20 * float(peak_val)

        # Walk backwards to the most recent sample below 20% of this peak.
        pre = envelope[: peak_idx + 1]
        below = np.flatnonzero(pre < threshold_20)
        start_idx = int(below[-1]) if below.size else 0
        rise_samples = max(0, peak_idx - start_idx)
        return 1000.0 * rise_samples / float(self.sample_rate)

    def _append_to_buffer(self, target, values):
        n = values.size
        if n >= self.buf_len:
            target[:] = values[-self.buf_len :]
        elif n > 0:
            target[:-n] = target[n:]
            target[-n:] = values

    def _analysis_channel(self, peak_l, peak_r):
        return self.buf_low_l if peak_l >= peak_r else self.buf_low_r

    def _update_side_evidence_locked(self, stereo_available, peak_l, peak_r):
        if not self.detect_sides or not stereo_available:
            return

        # Peak ratios are useful for chassis shock localization; log-space makes
        # left/right evidence symmetric. Clamp a single extreme tap so it cannot
        # dominate the whole multi-tap gesture.
        eps = 1e-8
        ratio = (peak_l + eps) / (peak_r + eps)
        log_ratio = float(np.clip(math.log(ratio), -math.log(4.0), math.log(4.0)))
        self._side_log_sum += log_ratio
        self._side_observations += 1

    def _group_side_locked(self):
        if not self.detect_sides or self._side_observations <= 0:
            return "center"

        avg_log = self._side_log_sum / self._side_observations
        boundary = math.log(self.side_ratio)
        if avg_log > boundary:
            return "left"
        if avg_log < -boundary:
            return "right"
        return "center"

    # ------------------------------------------------------------------
    # Main real-time path
    # ------------------------------------------------------------------

    def process_chunk(self, low_l, low_r, high_l, high_r, raw_chunk):
        if self._stopped:
            return

        try:
            low_l = np.asarray(low_l, dtype=np.float32).reshape(-1)
            low_r = np.asarray(low_r, dtype=np.float32).reshape(-1)
            high_l = np.asarray(high_l, dtype=np.float32).reshape(-1)
            high_r = np.asarray(high_r, dtype=np.float32).reshape(-1)
        except Exception:
            return

        step = min(low_l.size, low_r.size, high_l.size, high_r.size)
        if step <= 0:
            return

        if not (
            low_l.size == low_r.size == high_l.size == high_r.size
        ):
            # Mismatched filter outputs indicate an upstream bug. Use only the
            # common region instead of risking shape/broadcast failures.
            low_l = low_l[-step:]
            low_r = low_r[-step:]
            high_l = high_l[-step:]
            high_r = high_r[-step:]

        if not (
            np.isfinite(low_l).all()
            and np.isfinite(low_r).all()
            and np.isfinite(high_l).all()
            and np.isfinite(high_r).all()
        ):
            return

        now = time.monotonic()

        self._append_to_buffer(self.buf_low_l, low_l)
        self._append_to_buffer(self.buf_low_r, low_r)
        self._append_to_buffer(self.buf_high_l, high_l)
        self._append_to_buffer(self.buf_high_r, high_r)

        peak_l = float(np.max(np.abs(low_l)))
        peak_r = float(np.max(np.abs(low_r)))
        current_peak = max(peak_l, peak_r)
        threshold = self.calculate_threshold()

        callback = self.on_level_update
        if callback is not None and (now - self._last_level_emit >= 0.016):
            self._last_level_emit = now
            try:
                callback(current_peak, threshold)
            except Exception as exc:
                # UI/telemetry failures must not kill the audio callback.
                print(f"[TapDetector] Level callback error: {exc}")

        # Critical amplitude gate. The old implementation only returned below
        # 0.70*threshold, which allowed sub-threshold samples to become taps.
        if current_peak < threshold:
            self._update_noise_floor(current_peak, threshold)
            self._loud_duration_s = 0.0
            self.is_in_sustained_sound = False
            return

        self._loud_duration_s += step / float(self.sample_rate)

        # Suppression checks happen before the heavier feature extraction.
        if self.suppress_typing and (now - self.last_key_time) < self.typing_cooldown_s:
            self._record_candidate(
                "typing_suppressed", peak=current_peak, threshold=threshold
            )
            return
        if now < self.voice_speech_active_until:
            self._record_candidate(
                "speech_suppressed", peak=current_peak, threshold=threshold
            )
            return
        if now < self.voice_like_suppression_until:
            # Keep the latch alive until voice-like audio has actually been
            # quiet for the full hold period. Without this extension, a long
            # sentence can outlive the original latch and leak final syllables.
            self.voice_like_suppression_until = max(
                self.voice_like_suppression_until,
                now + self.voice_like_hold_s,
            )
            self._record_candidate(
                "voice_like_suppression", peak=current_peak, threshold=threshold
            )
            return
        if now < self.sustained_suppression_until:
            if self._loud_duration_s > self.sustained_revoke_s:
                self._discard_pending_group()
                self.voice_like_suppression_until = max(
                    self.voice_like_suppression_until,
                    now + self.voice_like_hold_s,
                )
            self._record_candidate(
                "sustained_recovery", peak=current_peak, threshold=threshold
            )
            return
        if now < self.post_gesture_suppression_until:
            self._record_candidate(
                "post_gesture_cooldown", peak=current_peak, threshold=threshold
            )
            return

        # Long above-threshold signals are much more likely to be speech/music,
        # HVAC/handling rumble, etc. Duration is measured in seconds rather than
        # "number of blocks", so it remains correct at 44.1/48/96 kHz and with
        # different block sizes.
        if self._loud_duration_s > self.max_sustained_s:
            self.is_in_sustained_sound = True
            self.sustained_suppression_until = max(
                self.sustained_suppression_until,
                now + self.sustained_recovery_s,
            )
            # A short chassis resonance may legitimately remain above the
            # sustained threshold. It is ignored here but the pending tap is
            # only revoked later if audio remains loud past sustained_revoke_s.
            self._update_noise_floor(current_peak, threshold, sustained=True)
            self._record_candidate(
                "sustained_sound", peak=current_peak, threshold=threshold,
                duration_ms=self._loud_duration_s * 1000.0,
            )
            return
        if self.is_in_sustained_sound:
            self._record_candidate(
                "sustained_sound", peak=current_peak, threshold=threshold,
                duration_ms=self._loud_duration_s * 1000.0,
            )
            return

        # High-frequency airborne rejection. Preserve the existing config's
        # energy-ratio semantics for backwards-compatible tuning.
        e_low = float(np.dot(low_l, low_l) + np.dot(low_r, low_r))
        if e_low <= 1e-12:
            self._record_candidate(
                "low_band_energy", peak=current_peak, threshold=threshold
            )
            return
        e_high = float(np.dot(high_l, high_l) + np.dot(high_r, high_r))
        hf_ratio = e_high / (e_low + 1e-12)
        if not math.isfinite(hf_ratio) or hf_ratio > self.max_hf_ratio:
            self._record_candidate(
                "high_frequency", peak=current_peak, threshold=threshold,
                hf_ratio=hf_ratio,
            )
            return

        # Crest factor over ~12 ms of history, independent of sample rate.
        win = max(64, min(self.buf_len, int(round(self.sample_rate * 0.012))))
        recent_low = self._analysis_channel(peak_l, peak_r)[-win:]
        win_peak = float(np.max(np.abs(recent_low)))
        win_rms = float(np.sqrt(np.mean(np.square(recent_low, dtype=np.float64))))
        crest_factor = win_peak / (win_rms + 1e-9)
        if not math.isfinite(crest_factor) or crest_factor < self.min_crest_factor:
            self._record_candidate(
                "low_crest", peak=current_peak, threshold=threshold,
                hf_ratio=hf_ratio, crest_factor=crest_factor,
            )
            return

        peak_idx = int(np.argmax(np.abs(recent_low)))
        rise_time_ms = self._compute_rise_time_ms(recent_low, peak_idx, win_peak)
        if rise_time_ms > self.max_rise_time_ms:
            self._record_candidate(
                "slow_rise", peak=current_peak, threshold=threshold,
                hf_ratio=hf_ratio, crest_factor=crest_factor,
                rise_time_ms=rise_time_ms,
            )
            return

        # Periodicity guard. Use the dominant channel and a duration-based
        # window so the pitch lag range remains meaningful at all sample rates.
        harm_win = max(
            int(self.sample_rate / 80.0) + 16,
            min(self.buf_len, int(round(self.sample_rate * 0.040))),
        )
        harm_win = min(harm_win, self.buf_len)
        harmonicity = self._compute_harmonicity(
            self._analysis_channel(peak_l, peak_r)[-harm_win:]
        )
        if harmonicity > self.max_harmonicity:
            self._record_candidate(
                "harmonic", peak=current_peak, threshold=threshold,
                hf_ratio=hf_ratio, crest_factor=crest_factor,
                rise_time_ms=rise_time_ms, harmonicity=harmonicity,
            )
            return

        # Register the tap and schedule race-safe grouping.
        try:
            raw = np.asarray(raw_chunk)
            stereo_available = raw.ndim >= 2 and raw.shape[1] >= 2
        except Exception:
            stereo_available = False

        refractory = False
        with self.lock:
            if self._stopped:
                return
            if now - self.last_tap_time < self.refractory_period_s:
                refractory = True
            else:
                self.last_tap_time = now
                self.current_taps += 1
                self.max_peak_in_group = max(self.max_peak_in_group, current_peak)
                self._update_side_evidence_locked(stereo_available, peak_l, peak_r)
                self.current_side = self._group_side_locked()

                self._timer_generation += 1
                token = self._timer_generation
                self._cancel_timers_locked()

                if self.current_taps == 3:
                    self.group_timer = threading.Timer(
                        self.triple_confirm_s,
                        self._dispatch_gesture,
                        args=(token,),
                    )
                    self.group_timer.daemon = True
                    self.group_timer.start()
                elif self.current_taps > 3:
                    self.reset_timer = threading.Timer(
                        0.25, self._reset_state, args=(token,)
                    )
                    self.reset_timer.daemon = True
                    self.reset_timer.start()
                else:
                    self.group_timer = threading.Timer(
                        self.grouping_window_s,
                        self._dispatch_gesture,
                        args=(token,),
                    )
                    self.group_timer.daemon = True
                    self.group_timer.start()

        metrics = {
            "peak": current_peak,
            "threshold": threshold,
            "hf_ratio": hf_ratio,
            "crest_factor": crest_factor,
            "rise_time_ms": rise_time_ms,
            "harmonicity": harmonicity,
        }
        if refractory:
            self._record_candidate("refractory", **metrics)
            return
        self._record_candidate("accepted_pulse", **metrics)

    # ------------------------------------------------------------------
    # Race-safe gesture timers
    # ------------------------------------------------------------------

    def _cancel_timers_locked(self):
        if self.group_timer is not None:
            try:
                self.group_timer.cancel()
            except Exception:
                pass
            self.group_timer = None

        if self.reset_timer is not None:
            try:
                self.reset_timer.cancel()
            except Exception:
                pass
            self.reset_timer = None

    def _invalidate_timers_locked(self):
        self._timer_generation += 1
        self._cancel_timers_locked()

    def _clear_group_locked(self):
        self.current_taps = 0
        self.current_side = "center"
        self.max_peak_in_group = 0.0
        self._side_log_sum = 0.0
        self._side_observations = 0

    def _discard_pending_group(self):
        with self.lock:
            if self.current_taps <= 0 and self.group_timer is None and self.reset_timer is None:
                return
            self._invalidate_timers_locked()
            self._clear_group_locked()

    def _reset_state(self, token=None):
        with self.lock:
            if token is not None and token != self._timer_generation:
                return
            rejected = self.current_taps
            self._invalidate_timers_locked()
            self._clear_group_locked()

        if rejected > 3:
            print(
                f"[TapDetector] Ignored burst of {rejected} pulses "
                "(continuous rumble/noise)."
            )

    def _dispatch_gesture(self, token=None):
        cancelled_by_voice = False
        spatially_ambiguous = False
        with self.lock:
            # cancel() cannot stop a Timer callback that has already begun.
            # Generation tokens guarantee an old callback cannot dispatch a new
            # group's state.
            if token is not None and token != self._timer_generation:
                return

            taps = self.current_taps
            side = self._group_side_locked()
            # True only when stereo evidence was actually collected for this
            # group. On a mono microphone every tap localizes to "center"; if
            # crowded mode rejected those, detection would dead-lock entirely.
            has_side_evidence = self._side_observations > 0
            peak = self.max_peak_in_group

            if time.monotonic() < self.voice_like_suppression_until:
                cancelled_by_voice = True
            if self.crowded_mode and side == "center" and has_side_evidence:
                spatially_ambiguous = True

            self._invalidate_timers_locked()
            self._clear_group_locked()

            if 1 <= taps <= 3:
                self.post_gesture_suppression_until = max(
                    self.post_gesture_suppression_until,
                    time.monotonic() + self.post_gesture_cooldown_s,
                )

        if cancelled_by_voice:
            self._record_candidate(
                "voice_like_group_cancelled", peak=peak, tap_count=taps
            )
        elif spatially_ambiguous:
            self._record_candidate(
                "spatially_ambiguous", peak=peak, tap_count=taps, side=side
            )
        elif 1 <= taps <= 3:
            gesture = DetectedTapGesture(count=taps, side=side, peak=peak)
            print(f"[TapDetector] >> GESTURE DETECTED: {gesture}")
            callback = self.on_gesture_detected
            if callback is not None:
                try:
                    callback(gesture)
                except Exception as exc:
                    print(f"[TapDetector] Gesture callback error: {exc}")
        elif taps > 3:
            print(f"[TapDetector] Rejected {taps} pulses (continuous noise)")

    def finalize_pending(self):
        """Finalize a pending gesture immediately (useful for deterministic tests)."""
        with self.lock:
            if self.current_taps <= 0:
                return False
            token = self._timer_generation
        self._dispatch_gesture(token)
        return True
