import queue
import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt


class AudioEngine:
    """Microphone capture with a minimal PortAudio callback and worker-thread DSP."""

    def __init__(
        self,
        sample_rate=48000,
        block_size=256,
        device_index=None,
        device_signature=None,
        on_audio_chunk=None,
        warmup_seconds=0.35,
    ):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.device_index = device_index
        self.device_signature = device_signature
        self.active_device_index = None
        self.active_device_signature = None
        self.active_device_name = None
        self.on_audio_chunk = on_audio_chunk
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self._warmup_frames_remaining = 0
        self.stream = None
        self.is_running = False
        self._lock = threading.RLock()
        self._history_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._generation = 0
        self._audio_queue = queue.Queue(maxsize=12)
        self._worker_thread = None
        self.last_status = None
        self.last_callback_error = None
        self.last_start_error = None
        self.last_device_change_error = None
        self.dropped_blocks = 0
        self.waveform_history_len = 1024
        self.waveform_history = np.zeros(self.waveform_history_len, dtype=np.float32)
        self._design_filters(self.sample_rate)

    def _design_filters(self, sample_rate):
        if sample_rate <= 2500:
            raise ValueError(f"Sample rate {sample_rate} Hz is too low for the 1200 Hz rejection band")
        nyquist = 0.5 * sample_rate
        self.sos_low = butter(2, [35.0 / nyquist, 450.0 / nyquist], btype="bandpass", output="sos")
        self.sos_high = butter(2, 1200.0 / nyquist, btype="highpass", output="sos")
        self.state_low_l = np.zeros((self.sos_low.shape[0], 2), dtype=np.float64)
        self.state_low_r = np.zeros((self.sos_low.shape[0], 2), dtype=np.float64)
        self.state_high_l = np.zeros((self.sos_high.shape[0], 2), dtype=np.float64)
        self.state_high_r = np.zeros((self.sos_high.shape[0], 2), dtype=np.float64)

    @staticmethod
    def _device_signature(index, dev=None):
        try:
            dev = dev or sd.query_devices(index, "input")
            hostapis = sd.query_hostapis()
            host_idx = int(dev.get("hostapi", -1))
            host_name = hostapis[host_idx].get("name", "unknown") if 0 <= host_idx < len(hostapis) else "unknown"
            return f"{dev.get('name', '')}|{host_name}".casefold()
        except Exception:
            return None

    @staticmethod
    def get_input_devices():
        devices = []
        try:
            hostapis = sd.query_hostapis()
            for idx, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) <= 0:
                    continue
                host_idx = int(dev.get("hostapi", -1))
                host_name = hostapis[host_idx].get("name", "Unknown") if 0 <= host_idx < len(hostapis) else "Unknown"
                devices.append(
                    {
                        "index": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "hostapi": host_name,
                        "signature": AudioEngine._device_signature(idx, dev),
                        "channels": int(dev.get("max_input_channels", 1)),
                        "default_samplerate": float(dev.get("default_samplerate", 48000)),
                    }
                )
        except Exception as exc:
            print(f"[AudioEngine] Error querying devices: {exc}")
        return devices

    @staticmethod
    def resolve_device_index(device_index=None, device_signature=None):
        devices = AudioEngine.get_input_devices()
        signature = str(device_signature or "").casefold()
        if signature:
            for dev in devices:
                if dev["signature"] == signature:
                    return dev["index"]
        if device_index is not None:
            try:
                device_index = int(device_index)
                if any(dev["index"] == device_index for dev in devices):
                    return device_index
            except (TypeError, ValueError):
                pass
        return None

    @staticmethod
    def find_internal_mic_index():
        devices = AudioEngine.get_input_devices()
        preferred_tokens = ("microphone array", "mic array", "array", "realtek")
        for token in preferred_tokens:
            for dev in devices:
                if token in dev["name"].lower() and "mapper" not in dev["name"].lower():
                    return dev["index"]
        try:
            default_in = sd.default.device[0]
            if default_in is not None and int(default_in) >= 0:
                return int(default_in)
        except Exception:
            pass
        return devices[0]["index"] if devices else None

    def _candidate_devices(self, allow_fallback=True):
        resolved = self.resolve_device_index(self.device_index, self.device_signature)
        if not allow_fallback:
            return [resolved] if resolved is not None else []

        devices = self.get_input_devices()
        candidates = []

        def is_generic(dev):
            name = dev["name"].lower()
            return any(token in name for token in ("mapper", "primary sound capture"))

        def score(dev):
            name = dev["name"].lower()
            host = dev["hostapi"].lower()
            value = 0
            if any(token in name for token in ("microphone array", "mic array", "realtek")):
                value += 100
            if "wasapi" in host:
                value += 30
            elif "directsound" in host:
                value += 20
            elif "mme" in host:
                value += 5
            if any(token in name for token in ("mapper", "primary", "stereo mix", "line")):
                value -= 80
            return value

        resolved_dev = next((dev for dev in devices if dev["index"] == resolved), None)
        # Windows mapper/primary aliases can hide the microphone-array channel
        # layout. Prefer the concrete hardware endpoint for knock localization.
        if resolved_dev is not None and not is_generic(resolved_dev):
            candidates.append(resolved)

        for dev in sorted(devices, key=score, reverse=True):
            idx = dev["index"]
            if idx not in candidates and not is_generic(dev) and score(dev) > -50:
                candidates.append(idx)
        if resolved is not None and resolved not in candidates:
            candidates.append(resolved)
        return candidates

    def _open_stream_for_device(self, dev_idx):
        dev_info = sd.query_devices(dev_idx, "input")
        max_channels = int(dev_info.get("max_input_channels", 0))
        if max_channels < 1:
            raise RuntimeError("Selected device has no input channels")
        actual_sr = int(round(float(dev_info.get("default_samplerate", self.sample_rate))))
        if actual_sr < 8000:
            actual_sr = self.sample_rate

        attempts = []
        channel_options = [2, 1] if max_channels >= 2 else [1]
        for channels in channel_options:
            for blocksize, latency in ((self.block_size, "low"), (self.block_size, None), (0, None)):
                try:
                    sd.check_input_settings(
                        device=dev_idx,
                        channels=channels,
                        samplerate=actual_sr,
                        dtype="float32",
                    )
                    kwargs = dict(
                        device=dev_idx,
                        channels=channels,
                        samplerate=actual_sr,
                        blocksize=blocksize,
                        dtype="float32",
                        callback=self._audio_callback,
                    )
                    if latency is not None:
                        kwargs["latency"] = latency
                    return sd.InputStream(**kwargs), dev_info, actual_sr, channels, blocksize
                except Exception as exc:
                    attempts.append(f"{channels}ch/block={blocksize}/latency={latency or 'default'}: {exc}")
        raise RuntimeError("; ".join(attempts))

    def start(self, allow_fallback=True):
        with self._lock:
            if self.is_running:
                return True
            worker = self._worker_thread
            if worker is not None and worker.is_alive():
                # Never clear a stop signal while a previous worker can still
                # be mutating the filter state or consuming this queue.
                worker.join()
            self._worker_thread = None
            self.last_start_error = None
            self.last_callback_error = None
            self.last_status = None
            self.dropped_blocks = 0
            errors = []

            for dev_idx in self._candidate_devices(allow_fallback=allow_fallback):
                try:
                    stream, dev_info, actual_sr, channels, blocksize = self._open_stream_for_device(dev_idx)
                    self.sample_rate = actual_sr
                    self._design_filters(actual_sr)
                    # Audio drivers and IIR filters can emit a short transient
                    # when a capture stream opens. Prime the DSP, history and
                    # filter state, but do not expose that transient as a tap.
                    self._warmup_frames_remaining = int(
                        round(actual_sr * self.warmup_seconds)
                    )
                    self._clear_audio_queue()
                    self._generation += 1
                    generation = self._generation
                    self._stop_event = threading.Event()
                    self.stream = stream
                    self.is_running = True
                    self.active_device_index = dev_idx
                    self.active_device_name = str(dev_info.get("name", f"Device {dev_idx}"))
                    self.active_device_signature = self._device_signature(dev_idx, dev_info)
                    self.device_index = dev_idx
                    self.device_signature = self.active_device_signature
                    worker = threading.Thread(
                        target=self._worker_loop,
                        args=(generation, self._stop_event),
                        name="OghdeiiAudioDSP",
                        daemon=True,
                    )
                    try:
                        stream.start()
                    except Exception:
                        self.is_running = False
                        self._stop_event.set()
                        stream.close()
                        self.stream = None
                        raise
                    self._worker_thread = worker
                    worker.start()
                    print(
                        f"[AudioEngine] Capturing device #{dev_idx} ({self.active_device_name}) "
                        f"at {actual_sr}Hz, {channels}ch, block={blocksize or 'auto'}"
                    )
                    return True
                except Exception as exc:
                    errors.append(f"#{dev_idx}: {exc}")

            self.is_running = False
            self.stream = None
            self.active_device_index = None
            detail = errors[-1] if errors else "No input device was found."
            self.last_start_error = (
                "Could not open a microphone. Check Windows microphone privacy settings and "
                f"close apps using exclusive access. Last attempt: {detail[-350:]}"
            )
            print(f"[AudioEngine] {self.last_start_error}")
            return False

    def _clear_audio_queue(self):
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                return

    def stop(self):
        with self._lock:
            stream, self.stream = self.stream, None
            self.is_running = False
            self._stop_event.set()
            self._generation += 1
            worker, self._worker_thread = self._worker_thread, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception as exc:
                print(f"[AudioEngine] Error closing audio stream: {exc}")
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        self._clear_audio_queue()
        if stream is not None:
            print("[AudioEngine] Audio stream stopped.")

    def change_device(self, new_device_index, new_device_signature=None):
        with self._lock:
            self.last_device_change_error = None
            old_index = self.device_index
            old_signature = self.device_signature
            was_running = self.is_running
            if was_running:
                self.stop()
            self.device_index = new_device_index
            self.device_signature = new_device_signature
            if not was_running:
                return True
            if self.start(allow_fallback=False):
                return True
            failure = self.last_start_error
            self.device_index = old_index
            self.device_signature = old_signature
            self.start()
            # The selected device failed, but fallback capture may be healthy.
            # Preserve the diagnostic separately from the active engine state.
            self.last_device_change_error = failure
            self.last_start_error = failure
            return False

    def get_waveform_history(self):
        with self._history_lock:
            return self.waveform_history.copy()

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback: copy and enqueue only; all DSP happens in the worker."""
        if not self.is_running or frames <= 0:
            return
        if status:
            self.last_status = str(status)
        try:
            chunk = np.asarray(indata, dtype=np.float32).copy()
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                self.dropped_blocks += 1
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass
        except Exception as exc:
            self.last_callback_error = repr(exc)

    def _worker_loop(self, generation, stop_event):
        while generation == self._generation and (
            not stop_event.is_set() or not self._audio_queue.empty()
        ):
            try:
                chunk = self._audio_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            try:
                self._process_chunk(chunk)
            except Exception as exc:
                self.last_callback_error = repr(exc)

    def _process_chunk(self, indata):
        if indata.ndim != 2 or indata.shape[0] <= 0 or indata.shape[1] <= 0:
            return
        channels = indata.shape[1]
        low_l, self.state_low_l = sosfilt(self.sos_low, indata[:, 0], zi=self.state_low_l)
        high_l, self.state_high_l = sosfilt(self.sos_high, indata[:, 0], zi=self.state_high_l)
        if channels >= 2:
            low_r, self.state_low_r = sosfilt(self.sos_low, indata[:, 1], zi=self.state_low_r)
            high_r, self.state_high_r = sosfilt(self.sos_high, indata[:, 1], zi=self.state_high_r)
        else:
            low_r = low_l.copy()
            high_r = high_l.copy()

        step = len(low_l)
        with self._history_lock:
            if step >= self.waveform_history_len:
                self.waveform_history[:] = low_l[-self.waveform_history_len :]
            elif step > 0:
                self.waveform_history[:-step] = self.waveform_history[step:]
                self.waveform_history[-step:] = low_l

        if self._warmup_frames_remaining > 0:
            self._warmup_frames_remaining = max(
                0, self._warmup_frames_remaining - step
            )
            return

        callback = self.on_audio_chunk
        if callback is not None:
            callback(low_l, low_r, high_l, high_r, indata)
