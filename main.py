import argparse
import sys
import time
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from action_executor import (
    ActionExecutor,
    resolve_gesture_action,
    resolve_voice_action,
)
from audio_engine import AudioEngine
from config_manager import ConfigManager
from gui import MainWindow
from oghdeii.resources import resource_path
from oghdeii.voice.v1m_verifier import V1MVoiceVerifier
from tap_detector import TapDetector
from voice_detector import VoiceDetector
from whisper_voice_detector import WhisperVoiceDetector


def acquire_single_instance_lock():
    """Ensure exactly one dispatcher process.

    Previously two instances (tray daemon + a second launch from the shortcut)
    could run side by side, each with its own in-memory config snapshot — the
    one you were NOT editing kept dispatching the old mappings, and whichever
    saved next clobbered the other's changes on disk.

    Returns the held QLockFile (must stay referenced for process lifetime) or
    None when another instance already owns the lock.
    """
    from PyQt6.QtCore import QLockFile

    lock_path = Path.home() / ".laptoptap" / "instance.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)  # first run: ConfigManager hasn't created it yet
    except OSError:
        pass
    lock = QLockFile(str(lock_path))
    lock.setStaleLockTime(30_000)  # also auto-clears if our PID died in a crash
    if lock.tryLock(0):
        return lock
    return None


def run_cli_mode(config, executor, detector, audio, voice, verifier):
    mode = config.get("active_mode", "knock")
    if mode not in ("knock", "voice"):
        mode = "knock"

    print("=====================================================")
    print("  عقده ای · حالت خط فرمان و پس‌زمینه")
    print("=====================================================")
    print(f"  * Active mode: {mode}")
    if mode == "knock":
        print("  * Knock 1 time:  ", config.get("actions", {}).get("single_tap"))
        print("  * Knock 2 times: ", config.get("actions", {}).get("double_tap"))
        print("  * Knock 3 times: ", config.get("actions", {}).get("triple_tap"))
    else:
        if config.get("voice_backend", "whisper") == "whisper":
            print(
                "  * Voice recognition: local faster-whisper "
                f"({config.get('whisper_model', 'distil-small.en')})"
            )
        else:
            print("  * Voice recognition: Windows System.Speech helper")
        print(f"  * v1m cloud guardrail: {'on' if verifier.enabled else 'off'}"
              f" (key: {verifier.key_source()})")
        print("  * v1m waiting timeout: "
              f"{config.get('v1m_wait_timeout_ms', 3000)} ms (blocked if V1M is enabled and times out)")
    print("  Press Ctrl+C to terminate.")
    print("-----------------------------------------------------")

    def on_gesture(gesture):
        # Same resolver the GUI uses: live config lookup, no restart needed.
        action_name = resolve_gesture_action(config, gesture)
        print(f"\n[CLI] Gesture: {gesture} -> {action_name}")
        executor.trigger(action_name)

    def on_voice(cmd, conf, text="", alternates=None):
        # Guardrail: one bounded system_one pass on a worker thread.
        # Every failure path returns an offline result, so this never blocks or
        # raises here — the recogniser reader keeps running either way.
        decision = verifier.verify(cmd, text, conf, alternates=alternates)
        if not decision.allow:
            print(f"\n[CLI] v1m blocked '{text or cmd}': {decision.reason}")
            return
        action_name = resolve_voice_action(config, decision.command)
        if not action_name:
            print(f"[CLI] Ignoring unmapped voice command: {decision.command}")
            return
        disp = text or cmd
        guard = ""
        if decision.is_cloud:
            guard = f" [v1m {decision.probability:.0%}, {decision.source}]"
        print(f"\n[CLI] Voice: '{disp}' ({conf * 100:.0f}%){guard} -> {action_name}")
        executor.trigger(action_name)

    detector.on_gesture_detected = on_gesture
    voice.on_voice_command = on_voice

    started = False
    try:
        if mode == "knock":
            detector.start()
            started = audio.start()
            if started:
                config.update({
                    "selected_input_device": audio.active_device_index,
                    "selected_input_device_signature": audio.active_device_signature,
                })
        elif not config.get("enable_voice_commands", True):
            print("[CLI] Voice mode is selected, but voice commands are disabled in config.")
            return 2
        else:
            started = voice.start()
            if started:
                ready_timeout = 300.0 if config.get("voice_backend", "whisper") == "whisper" else 8.0
                started = voice.wait_until_ready(ready_timeout)

        if not started:
            detail = voice.last_error if mode == "voice" else audio.last_start_error
            print(f"[CLI] Failed to start {mode} engine: {detail or 'unknown error'}")
            return 1

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[CLI] Stopping engines...")
    finally:
        audio.stop()
        if hasattr(voice, "cleanup"):
            voice.cleanup()
        else:
            voice.stop()
        detector.stop()
        verifier.shutdown()
        config.flush()
    print("[CLI] Exited cleanly.")
    return 0


def main():
    # Redirected Windows output often defaults to a legacy code page that
    # cannot encode the Persian app name, including in --help and errors.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="عقده ای — میانبرهای ضربه و صدا")
    parser.add_argument("--cli", action="store_true", help="Run in headless console mode without GUI")
    parser.add_argument("--minimized", "--tray", action="store_true", help="Start minimized to system tray")
    args = parser.parse_args()

    lock = acquire_single_instance_lock()
    if lock is None:
        if args.cli:
            print("[عقده ای] برنامه از قبل در حال اجراست؛ نمونه‌ی دوم شروع نمی‌شود.")
            return 1
        if not args.minimized:
            app = QApplication(sys.argv)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                None,
                "عقده ای",
                "برنامه از قبل در حال اجراست.\n\n"
                "برای تنظیم میانبرها، آن را از system tray باز کن.",
            )
        else:
            print("[عقده ای] برنامه از قبل در حال اجراست؛ این اجرا بسته می‌شود.")
        return 0

    config = ConfigManager()
    executor = ActionExecutor(config_manager=config)
    detector = TapDetector(config_manager=config)
    audio = AudioEngine(
        device_index=config.get("selected_input_device"),
        device_signature=config.get("selected_input_device_signature"),
    )

    def handle_audio_chunk(ll, lr, hl, hr, raw):
        # AudioEngine may adopt the microphone's native 44.1/48/96 kHz rate.
        # Keep TapDetector's timing and pitch analysis synchronized.
        detector.set_sample_rate(audio.sample_rate)
        detector.process_chunk(ll, lr, hl, hr, raw)

    audio.on_audio_chunk = handle_audio_chunk
    voice_backend = config.get("voice_backend", "whisper")
    if voice_backend == "windows" and sys.platform != "win32":
        print("[Voice] Windows Speech helper is unavailable; falling back to local Whisper.")
        config.set("voice_backend", "whisper")
        voice_backend = "whisper"
    elif voice_backend == "whisper":
        try:
            import faster_whisper  # noqa: F401
        except Exception:
            if sys.platform == "win32" and (resource_path("VoiceRecognizerV1M.exe")).exists():
                print("[Voice] faster-whisper is not installed; falling back to Windows Speech helper.")
                config.set("voice_backend", "windows")
                voice_backend = "windows"
    voice_class = WhisperVoiceDetector if voice_backend == "whisper" else VoiceDetector
    voice = voice_class(config_manager=config)
    # Cloud guardrail for voice commands: no-ops (offline fallback) unless the
    # toggle is on and V1M_API_KEY is present.
    verifier = V1MVoiceVerifier(config)
    verifier.start()

    if args.cli:
        return run_cli_mode(config, executor, detector, audio, voice, verifier)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow(
        config, audio, detector, executor,
        voice_detector=voice, voice_verifier=verifier,
    )
    # The window owns the current backend, which can change in Settings.
    app.aboutToQuit.connect(window.clean_quit)
    if not args.minimized:
        window.show()
    else:
        print("[عقده ای] برنامه در system tray شروع شد.")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
