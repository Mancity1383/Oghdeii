import base64
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from config_manager import ConfigManager

CALIBRATION_PHRASES = {
    "copy": "copy that",
    "paste": "paste that",
    "undo": "undo that",
    "screenshot": "take screenshot",
}

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

def play_beep(success=True):
    if not HAS_WINSOUND:
        return
    try:
        winsound.Beep(1200 if success else 450, 70 if success else 100)
    except Exception:
        pass


def make_meter(val, max_len=10):
    val = max(0.0, min(1.0, val))
    filled = int(round(val * max_len))
    return "#" * filled + "-" * (max_len - filled)


class VoiceCalibrator:
    def __init__(self):
        self.config = ConfigManager()
        self.exe_path = Path(__file__).parent / "VoiceRecognizerV1M.exe"
        if os.name != "nt":
            raise RuntimeError("Voice calibration requires Windows System.Speech.")
        if not self.exe_path.exists():
            raise FileNotFoundError(f"VoiceRecognizerV1M.exe was not found at: {self.exe_path}")

    def _start_recognizer(self):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        mode = "wake" if self.config.get("voice_require_wake_word", True) else "direct"
        return subprocess.Popen(
            [str(self.exe_path), "0.01", mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=startupinfo,
        )

    @staticmethod
    def _parse_event(line):
        """Return ``(kind, canonical, confidence, spoken)`` or None.

        ``kind`` is the raw protocol line kind. ``DEBUG_REJECTED`` entries are
        diagnostics — Windows matched the phrase but refused it — so the
        callers report them and never count them as captured samples or as
        false candidates, matching how voice_validate.py scores them.
        """
        if line.strip().startswith("VOICE_TEXT2:"):
            parts = line.strip().split(":", 2)
            if len(parts) != 3:
                return None
            try:
                conf = float(parts[1].replace(",", "."))
                payload = base64.b64decode(parts[2], validate=True).decode("utf-8")
                first = payload.split("\x1f", 1)[0]
                _candidate_conf, spoken = first.split("\x1e", 1)
            except (ValueError, TypeError, UnicodeDecodeError):
                return None
            spoken = spoken.strip().lower()
            if not 0.0 <= conf <= 1.0 or not spoken:
                return None
            return "VOICE_TEXT", spoken.replace(" ", "_"), conf, spoken

        if line.strip().startswith("VOICE_TEXT:"):
            parts = line.strip().split(":", 2)
            if len(parts) != 3:
                return None
            try:
                conf = float(parts[1].replace(",", "."))
            except ValueError:
                return None
            spoken = parts[2].strip().lower()
            if not 0.0 <= conf <= 1.0 or not spoken:
                return None
            return "VOICE_TEXT", spoken.replace(" ", "_"), conf, spoken

        parts = line.strip().split(":", 3)
        if len(parts) < 3 or parts[0] not in ("VOICE_CMD", "DEBUG_LOW_CONF", "DEBUG_REJECTED"):
            return None
        try:
            conf = float(parts[2].replace(",", "."))
        except ValueError:
            return None
        if not 0.0 <= conf <= 1.0:
            return None
        canonical = parts[1].strip().lower()
        spoken = parts[3].strip().lower() if len(parts) >= 4 else canonical
        return parts[0], canonical, conf, spoken

    @staticmethod
    def _reader_thread(stream, out_queue):
        try:
            for line in stream:
                out_queue.put(line)
        finally:
            out_queue.put(None)

    @staticmethod
    def _helper_already_running():
        """True when the voice recognizer helper already owns the microphone
        (LaptopTap in voice mode, or a leftover helper).  Spawning a second
        one makes both fight over the default input device and corrupts each
        other's stdout lines."""
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq VoiceRecognizerV1M.exe", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return "voicerecognizerv1m.exe" in (out.stdout or "").lower()
        except Exception:
            return False  # tasklist unavailable: don't block calibration

    def _start_ready_recognizer(self, timeout_s=8.0):
        if self._helper_already_running():
            raise RuntimeError(
                "The voice recognizer helper is already running (LaptopTap voice mode or "
                "a leftover helper). Exit LaptopTap from the tray — or end the "
                "the helper process — before calibrating."
            )
        proc = self._start_recognizer()
        output = queue.Queue()
        threading.Thread(target=self._reader_thread, args=(proc.stdout, output), daemon=True).start()
        deadline = time.monotonic() + timeout_s
        statuses = []
        while time.monotonic() < deadline:
            try:
                line = output.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            line = line.strip()
            if line.startswith("STATUS:"):
                statuses.append(line)
                if line == "STATUS:READY":
                    return proc, output
            elif line.startswith("ERROR:"):
                self._safe_close_proc(proc)
                raise RuntimeError(line[6:].strip())
        self._safe_close_proc(proc)
        detail = statuses[-1] if statuses else "no status received"
        raise RuntimeError(f"Voice engine did not become ready ({detail}).")

    def _safe_close_proc(self, proc):
        try:
            if proc.poll() is None and proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                # Closing our write end guarantees stdin EOF reaches the
                # helper: even if "quit" was lost, its ReadLine() returns
                # null and the child exits instead of being orphaned.
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        except Exception:
            pass

    def capture_command_trials(self, command_name, target_count=5, timeout_s=30):
        print("\n" + "-" * 66)
        print(f"  COMMAND: {command_name.upper()}")
        print("-" * 66)
        command_phrase = CALIBRATION_PHRASES.get(command_name, command_name)
        phrase = f"laptop {command_phrase}" if self.config.get("voice_require_wake_word", True) else command_phrase
        print(f"  Say '{phrase}' {target_count} times, naturally and separately.")
        input("  Press Enter to start listening... ")

        print("  Starting voice engine...", end="", flush=True)
        proc, q = self._start_ready_recognizer()
        print(" ready.")
        samples = []
        deadline = time.monotonic() + timeout_s
        try:
            while len(samples) < target_count and time.monotonic() < deadline:
                try:
                    line = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line.startswith("ERROR:"):
                    print(f"  Engine error: {line.strip()}")
                    break
                event = self._parse_event(line)
                if not event:
                    continue
                kind, canonical, conf, spoken = event
                if kind == "DEBUG_REJECTED":
                    # Refused by Windows after a grammar match: shown for
                    # diagnostics only. Counting it as a captured sample would
                    # drag the chosen confidence threshold downwards.
                    print(f"  ✗ Windows rejected '{spoken}' as {canonical} ({conf*100:.0f}%)")
                    continue
                if canonical == command_name or command_name in spoken:
                    samples.append(conf)
                    play_beep(True)
                    print(f"  PASS {len(samples)}/{target_count}  {conf*100:5.1f}% "
                          f"[{make_meter(conf)}]  heard='{spoken}'")
                elif conf >= 0.35:
                    print(f"  Heard '{spoken}' ({conf*100:.0f}%), waiting for '{command_name}'.")
        except KeyboardInterrupt:
            print(f"\n  Stage skipped; {len(samples)} sample(s) kept.")
        finally:
            self._safe_close_proc(proc)
        return samples

    def capture_negative_trials(self, duration_s=10):
        print("\n" + "-" * 66)
        print("  FALSE-POSITIVE / AMBIENT TEST")
        print("-" * 66)
        print(
            "  Do not say any configured command or wake phrase. Normal speech, "
            "typing, and ordinary room noise are allowed."
        )
        input("  Press Enter to start the negative test... ")
        print("  Starting voice engine...", end="", flush=True)
        proc, q = self._start_ready_recognizer()
        print(" ready.")
        negatives = []
        end = time.monotonic() + duration_s
        try:
            while time.monotonic() < end:
                try:
                    line = q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line is None:
                    break
                event = self._parse_event(line)
                if event:
                    kind, canonical, conf, spoken = event
                    if kind == "DEBUG_REJECTED":
                        # Declined by Windows: reported, but not a false
                        # candidate — the engine already refused it.
                        print(f"  · rejected '{spoken}' ({conf*100:.0f}%)")
                        continue
                    negatives.append(conf)
                    print(f"  ! false candidate: '{spoken}' -> {canonical} ({conf*100:.1f}%)")
        finally:
            self._safe_close_proc(proc)
        if not negatives:
            print("  PASS: no false candidate was recorded.")
        return negatives

    @staticmethod
    def choose_threshold(positives, negatives):
        """Choose a conservative threshold, heavily penalizing false accepts."""
        if not positives:
            return 0.55, None

        candidates = [round(x / 100.0, 2) for x in range(40, 96)]
        best = None
        for t in candidates:
            tpr = sum(c >= t for c in positives) / len(positives)
            fpr = (sum(c >= t for c in negatives) / len(negatives)) if negatives else 0.0
            # Accuracy goal: false triggers are costlier than an occasional missed command.
            score = tpr - 2.5 * fpr
            # Prefer higher threshold on ties for precision.
            key = (score, -fpr, tpr, t)
            if best is None or key > best[0]:
                best = (key, t, tpr, fpr)

        _, threshold, tpr, fpr = best
        # Never choose an overly permissive threshold solely because calibration was too easy.
        threshold = max(0.55, min(0.90, threshold))
        return threshold, (tpr, fpr)

    def run_full_calibration(self):
        print("\n" + "=" * 66)
        print("       LAPTOPTAP VOICE CALIBRATION — PRECISION-FIRST")
        print("=" * 66)
        print("  No actions are executed. This measures commands and ambient false positives.")
        print("  IMPORTANT: Exit the normal LaptopTap app from the system tray first.")

        commands = ["copy", "paste", "undo", "screenshot"]
        results = {}
        for cmd in commands:
            vals = self.capture_command_trials(cmd, target_count=5)
            if vals:
                results[cmd] = vals
            else:
                print(f"  WARNING: no sample was recorded for {cmd}.")

        positives = [v for values in results.values() for v in values]
        if not positives:
            print("\nNo command samples were recorded; settings were not changed.")
            return

        negatives = self.capture_negative_trials(duration_s=20)
        threshold, metrics = self.choose_threshold(positives, negatives)

        print("\n" + "=" * 66)
        print("  RESULTS")
        print("=" * 66)
        for cmd in commands:
            vals = results.get(cmd, [])
            if vals:
                print(
                    f"  {cmd.upper():<12} avg={sum(vals)/len(vals)*100:5.1f}% "
                    f"min={min(vals)*100:5.1f}% max={max(vals)*100:5.1f}% n={len(vals)}"
                )
        print(f"  Positive samples: {len(positives)}")
        print(f"  False candidates: {len(negatives)}")
        if metrics:
            tpr, fpr = metrics
            print(f"  Estimated command acceptance at candidate threshold: {tpr*100:.1f}%")
            if negatives:
                print(f"  Estimated false-candidate acceptance: {fpr*100:.1f}%")
        print(f"  Selected threshold: {threshold:.2f} ({threshold*100:.0f}%)")
        print("  These measurements describe this session; they are not a 100% guarantee.")

        self.config.set("voice_confidence_threshold", round(threshold, 2))
        self.config.set("enable_voice_commands", True)
        play_beep(True)
        print("\n  PASS: the calibrated confidence threshold was saved.")


if __name__ == "__main__":
    try:
        VoiceCalibrator().run_full_calibration()
    except KeyboardInterrupt:
        print("\nCalibration cancelled by user.")
    except Exception as exc:
        print(f"\nCalibration failed: {exc}")
        sys.exit(1)
