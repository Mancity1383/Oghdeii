"""Interactive, action-free hardware validation for LaptopTap's knock detector."""

import argparse
import json
import queue
import sys
import time
from collections import Counter
from pathlib import Path

from audio_engine import AudioEngine
from config_manager import ConfigManager
from tap_detector import TapDetector


def _drain(items):
    drained = []
    while True:
        try:
            drained.append(items.get_nowait())
        except queue.Empty:
            return drained


def _wait(seconds, label):
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sys.stdout.write(f"\r{label}: {remaining:4.1f}s  ")
        sys.stdout.flush()
        time.sleep(min(0.1, remaining))
    print(f"\r{label}: done.      ")


def _diagnostic_delta(before, after):
    keys = set(before) | set(after)
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in keys
        if after.get(key, 0) > before.get(key, 0)
    }


def _accepted_candidates(records, start_index):
    """Keep accepted pulse features so failed trials can be tuned from the report."""
    fields = (
        "timestamp", "peak", "threshold", "hf_ratio", "crest_factor",
        "rise_time_ms", "harmonicity",
    )
    return [
        {key: record[key] for key in fields if key in record}
        for record in records[start_index:]
        if record.get("result") == "accepted_pulse"
    ]


def _relevant_rejections(records, start_index, limit=200):
    """Retain bounded rejection timing/features for the next tuning pass."""
    ignored = {"accepted_pulse", "typing_suppressed", "sustained_sound"}
    fields = (
        "result", "timestamp", "peak", "threshold", "hf_ratio",
        "crest_factor", "rise_time_ms", "harmonicity", "duration_ms",
    )
    selected = [
        {key: record[key] for key in fields if key in record}
        for record in records[start_index:]
        if record.get("result") not in ignored
    ]
    return selected[-limit:]


def run_validation(
    trials=5,
    negative_seconds=10.0,
    report_path=None,
    speech_only=False,
    chatter_only=False,
    taps_only=False,
):
    # Keep output usable in legacy Windows CMD code pages as well as UTF-8 terminals.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")

    config = ConfigManager()
    gestures = queue.Queue()
    candidate_records = []
    detector = TapDetector(
        config,
        on_gesture_detected=gestures.put,
        monitor_keyboard=True,
        on_candidate_evaluated=candidate_records.append,
    )
    audio = AudioEngine(
        device_index=config.get("selected_input_device"),
        device_signature=config.get("selected_input_device_signature"),
    )

    def process_audio(low_l, low_r, high_l, high_r, raw):
        detector.set_sample_rate(audio.sample_rate)
        detector.process_chunk(low_l, low_r, high_l, high_r, raw)

    audio.on_audio_chunk = process_audio
    detector.start()

    print("=" * 70)
    print("LaptopTap - Real Chassis Tap Validation (no actions are executed)")
    print("=" * 70)
    print("IMPORTANT: Exit the normal LaptopTap app from the system tray first.")
    print("Tap the laptop chassis near the left or right side of the touchpad.")
    print("Use your knuckle and do not tap the keyboard keys.")

    if not audio.start():
        detector.stop()
        print(f"\nERROR: {audio.last_start_error}")
        print("Check Windows Settings > Privacy & security > Microphone.")
        return 2

    report_path = Path(report_path or "tap_validation_report.json").resolve()
    report = {
        "schema_version": 1,
        "started_at": time.time(),
        "device": {
            "index": audio.active_device_index,
            "name": audio.active_device_name,
            "signature": audio.active_device_signature,
            "sample_rate": audio.sample_rate,
        },
        "settings": {
            key: config.get(key)
            for key in (
                "sensitivity", "grouping_window_ms", "refractory_period_ms",
                "suppress_typing", "typing_cooldown_ms", "max_hf_ratio",
                "min_crest_factor", "max_rise_time_ms", "max_harmonicity",
                "max_sustained_ms", "detect_sides", "side_ratio",
                "crowded_mode",
                "sustained_revoke_ms", "sustained_recovery_ms",
                "voice_like_hold_ms", "post_gesture_cooldown_ms",
                "triple_confirm_ms",
            )
        },
        "negative_stages": [],
        "positive_trials": [],
    }

    cooldown = max(
        0.65,
        config.get("typing_cooldown_ms", 400) / 1000.0 + 0.20,
    )

    try:
        print(f"\nActive microphone: {audio.active_device_name} ({audio.sample_rate} Hz)")
        print("After you press Enter, the test pauses briefly so that key press is ignored.")

        if taps_only:
            negative_stages = ()
        elif chatter_only:
            negative_stages = (
                (
                    "chatter",
                    "Do not speak or touch the laptop. Leave it exposed to normal room chatter and background noise.",
                ),
            )
        elif speech_only:
            negative_stages = (
                ("speech", "Speak normally near the laptop, but do not touch it."),
            )
        else:
            negative_stages = (
                ("quiet", "Stay quiet. Do not touch the laptop or desk."),
                ("speech", "Speak normally near the laptop, but do not touch it."),
                ("typing", "Type continuously and normally on the laptop keyboard."),
            )
        for name, instruction in negative_stages:
            input(f"\n[NEGATIVE TEST: {name.upper()}] {instruction}\nPress Enter to begin...")
            _wait(cooldown, "Preparing")
            detector.reset()
            _drain(gestures)
            before = detector.get_diagnostics()["counts"]
            candidate_start = len(candidate_records)
            stage_start = time.monotonic()
            _wait(negative_seconds, "Listening")
            print("Stop this activity and remain still while pending audio settles.")
            settle_s = config.get("grouping_window_ms", 408) / 1000.0 + 0.20
            _wait(settle_s, "Settling")
            detector.finalize_pending()
            observed = _drain(gestures)
            after = detector.get_diagnostics()["counts"]
            stage = {
                "name": name,
                "duration_s": time.monotonic() - stage_start,
                "false_gestures": [gesture.name() for gesture in observed],
                "candidate_counts": _diagnostic_delta(before, after),
                "accepted_candidates": _accepted_candidates(
                    candidate_records, candidate_start
                ),
                "relevant_rejections": _relevant_rejections(
                    candidate_records, candidate_start
                ),
            }
            report["negative_stages"].append(stage)
            print(f"Result: {len(observed)} false detection(s)")

        listen_s = max(
            2.0,
            config.get("grouping_window_ms", 408) / 1000.0 + 1.35,
        )
        negative_only = speech_only or chatter_only
        for expected in (() if negative_only else (1, 2, 3)):
            for trial in range(1, trials + 1):
                input(
                    f"\n[{expected}-TAP TEST - {trial}/{trials}] Press Enter. "
                    f"After the 'TAP NOW' message, tap exactly {expected} time(s)..."
                )
                _wait(cooldown, "Preparing")
                detector.reset()
                _drain(gestures)
                before = detector.get_diagnostics()["counts"]
                candidate_start = len(candidate_records)
                print("*** TAP NOW ***")
                _wait(listen_s, "Listening")
                detector.finalize_pending()
                observed = _drain(gestures)
                after = detector.get_diagnostics()["counts"]
                counts = [gesture.count for gesture in observed]
                passed = len(counts) == 1 and counts[0] == expected
                report["positive_trials"].append(
                    {
                        "expected_count": expected,
                        "trial": trial,
                        "observed_counts": counts,
                        "observed_sides": [gesture.side for gesture in observed],
                        "passed": passed,
                        "candidate_counts": _diagnostic_delta(before, after),
                        "accepted_candidates": _accepted_candidates(
                            candidate_records, candidate_start
                        ),
                        "relevant_rejections": _relevant_rejections(
                            candidate_records, candidate_start
                        ),
                        "last_candidate": detector.get_diagnostics()["last_candidate"],
                    }
                )
                if passed:
                    print("[PASS] Correct gesture detected")
                else:
                    observed_text = counts if counts else "none"
                    print(f"[FAIL] Incorrect result. Detected: {observed_text}")

        positives = report["positive_trials"]
        passed = sum(1 for trial in positives if trial["passed"])
        false_gestures = sum(
            len(stage["false_gestures"]) for stage in report["negative_stages"]
        )
        negative_minutes = sum(
            stage["duration_s"] for stage in report["negative_stages"]
        ) / 60.0
        accuracy = passed / len(positives) if positives else 0.0
        false_per_minute = (
            false_gestures / negative_minutes if negative_minutes else 0.0
        )
        all_reasons = Counter()
        for stage in report["negative_stages"]:
            all_reasons.update(stage["candidate_counts"])
        for trial in positives:
            all_reasons.update(trial["candidate_counts"])

        quality_gate_passed = (
            false_gestures == 0
            and audio.dropped_blocks == 0
            and not audio.last_callback_error
            and (negative_only or accuracy >= 0.90)
        )
        report["summary"] = {
            "mode": (
                "chatter_only" if chatter_only
                else "speech_only" if speech_only
                else "taps_only" if taps_only
                else "full"
            ),
            "passed_trials": passed,
            "total_trials": len(positives),
            "gesture_accuracy": accuracy,
            "false_gestures": false_gestures,
            "negative_minutes": negative_minutes,
            "false_gestures_per_minute": false_per_minute,
            "dropped_audio_blocks": audio.dropped_blocks,
            "audio_callback_error": audio.last_callback_error,
            "candidate_counts": dict(all_reasons),
            "quality_gate_passed": quality_gate_passed,
        }
        report["finished_at"] = time.time()
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 70)
        if positives:
            print(f"Gesture accuracy: {passed}/{len(positives)} = {accuracy * 100:.1f}%")
        else:
            print("Positive gesture tests: skipped")
        print(f"False detections: {false_gestures} ({false_per_minute:.2f} per minute)")
        print(f"Dropped audio blocks: {audio.dropped_blocks}")
        gate = "PASS" if report["summary"]["quality_gate_passed"] else "NEEDS TUNING"
        print(f"Quality gate: {gate}")
        print(f"Report saved to: {report_path}")
        return 0 if report["summary"]["quality_gate_passed"] else 1
    except KeyboardInterrupt:
        print("\nValidation cancelled by user.")
        return 130
    finally:
        audio.stop()
        detector.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Validate LaptopTap against real laptop-chassis taps"
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="trials for each of 1/2/3 taps",
    )
    parser.add_argument(
        "--negative-seconds", type=float, default=10.0,
        help="seconds per quiet/speech/typing stage",
    )
    parser.add_argument(
        "--report", default="tap_validation_report.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--speech-only", action="store_true",
        help="run only the speech false-positive test",
    )
    parser.add_argument(
        "--chatter-only", action="store_true",
        help="run only a background room-chatter false-positive test",
    )
    parser.add_argument(
        "--taps-only", action="store_true",
        help="run only the positive 1/2/3-tap tests",
    )
    args = parser.parse_args()
    if not 1 <= args.trials <= 30:
        parser.error("--trials must be between 1 and 30")
    if not 2.0 <= args.negative_seconds <= 300.0:
        parser.error("--negative-seconds must be between 2 and 300")
    exclusive_modes = sum(
        bool(value)
        for value in (args.speech_only, args.chatter_only, args.taps_only)
    )
    if exclusive_modes > 1:
        parser.error(
            "--speech-only, --chatter-only, and --taps-only are mutually exclusive"
        )
    return run_validation(
        args.trials,
        args.negative_seconds,
        args.report,
        speech_only=args.speech_only,
        chatter_only=args.chatter_only,
        taps_only=args.taps_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
