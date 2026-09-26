"""Interactive, action-free validation for LaptopTap voice recognition."""

import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from config_manager import ConfigManager

ROOT = Path(__file__).resolve().parent
HELPER = ROOT / "VoiceRecognizerV1M.exe"
REPORT = ROOT / "voice_validation_report.json"
DEFAULT_COMMANDS = (
    ("copy", "copy that"),
    ("paste", "paste that"),
    ("undo", "undo that"),
    ("screenshot", "take screenshot"),
)


def reader(stream, output):
    try:
        for line in stream:
            output.put(line.strip())
    finally:
        output.put(None)


def parse_event(line):
    if line.startswith("VOICE_TEXT2:"):
        parts = line.split(":", 2)
        if len(parts) != 3:
            return None
        try:
            confidence = float(parts[1].replace(",", "."))
            payload = base64.b64decode(parts[2], validate=True).decode("utf-8")
            first = payload.split("\x1f", 1)[0]
            _candidate_conf, spoken = first.split("\x1e", 1)
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
        spoken = spoken.strip()
        if not 0.0 <= confidence <= 1.0 or not spoken:
            return None
        return "VOICE_TEXT", spoken.lower().replace(" ", "_"), confidence, spoken

    if line.startswith("VOICE_TEXT:"):
        parts = line.split(":", 2)
        if len(parts) != 3:
            return None
        try:
            confidence = float(parts[1].replace(",", "."))
        except ValueError:
            return None
        spoken = parts[2].strip()
        if not 0.0 <= confidence <= 1.0 or not spoken:
            return None
        return "VOICE_TEXT", spoken.lower().replace(" ", "_"), confidence, spoken

    parts = line.split(":", 3)
    if len(parts) < 3 or parts[0] not in (
        "VOICE_CMD", "DEBUG_LOW_CONF", "DEBUG_REJECTED"
    ):
        return None
    try:
        confidence = float(parts[2].replace(",", "."))
    except ValueError:
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    spoken = parts[3].strip() if len(parts) == 4 else parts[1].strip()
    return parts[0], parts[1].strip().lower(), confidence, spoken


def start_helper(mode):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    proc = subprocess.Popen(
        [str(HELPER), "0.01", mode],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        startupinfo=startupinfo,
    )
    output = queue.Queue()
    threading.Thread(target=reader, args=(proc.stdout, output), daemon=True).start()
    return proc, output


def stop_helper(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None and proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
            except Exception:
                pass
            # EOF fallback: guarantees the helper exits even if "quit" was lost.
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.wait(timeout=2.5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_ready(proc, output, timeout=8.0):
    deadline = time.monotonic() + timeout
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
        if line.startswith("STATUS:"):
            statuses.append(line)
            if line == "STATUS:READY":
                return True, statuses, None
        elif line.startswith("ERROR:"):
            return False, statuses, line[6:].strip()
    return False, statuses, "Voice engine did not become ready within 8 seconds."


def drain(output):
    while True:
        try:
            output.get_nowait()
        except queue.Empty:
            return


def collect_one(output, expected, timeout):
    deadline = time.monotonic() + timeout
    candidates = []
    best_rejected = None
    while time.monotonic() < deadline:
        try:
            line = output.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            break
        event = parse_event(line)
        if not event:
            continue
        kind, command, confidence, spoken = event
        candidates.append({
            "kind": kind,
            "command": command,
            "confidence": confidence,
            "spoken": spoken,
        })
        expected_text = expected.replace("_", " ").lower()
        spoken_normalized = " " + " ".join(spoken.lower().split()) + " "
        matches = command == expected or (
            kind == "VOICE_TEXT" and f" {expected_text} " in spoken_normalized
        )
        if matches and kind != "DEBUG_REJECTED":
            return candidates[-1], candidates
        if matches and (
            best_rejected is None or confidence > best_rejected["confidence"]
        ):
            best_rejected = candidates[-1]
    return best_rejected, candidates


def collect_negative(output, seconds):
    deadline = time.monotonic() + seconds
    events = []
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        print(f"\rListening: {remaining:4.1f}s", end="", flush=True)
        try:
            line = output.get(timeout=min(0.2, max(0.01, remaining)))
        except queue.Empty:
            continue
        if line is None:
            break
        event = parse_event(line)
        if event:
            kind, command, confidence, spoken = event
            events.append({
                "kind": kind,
                "command": command,
                "confidence": confidence,
                "spoken": spoken,
            })
    print("\rListening: done. ")
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Validate LaptopTap voice recognition without executing actions."
    )
    parser.add_argument("--trials", type=int, default=3, help="Attempts per command (default: 3)")
    parser.add_argument("--listen-seconds", type=float, default=7.0, help="Seconds per attempt")
    parser.add_argument("--negative-seconds", type=float, default=20.0, help="Ambient test duration")
    parser.add_argument("--startup-only", action="store_true", help="Only verify engine startup")
    parser.add_argument("--direct", action="store_true", help="Test commands without the Laptop wake word")
    args = parser.parse_args()

    if os.name != "nt":
        print("This validator requires Windows.")
        return 2
    if not HELPER.exists():
        print(f"Voice helper not found: {HELPER}")
        return 2

    config = ConfigManager()
    wake_required = False if args.direct else bool(config.get("voice_require_wake_word", True))
    mode = "wake" if wake_required else "direct"
    threshold = float(config.get("voice_confidence_threshold", 0.55))

    print("=" * 70)
    print("LaptopTap - Real Voice Validation (no actions are executed)")
    print("=" * 70)
    print("IMPORTANT: Exit the normal LaptopTap app from the system tray first.")
    print("Voice input uses the Windows default microphone.")
    print(f"Mode: {mode.upper()} | configured confidence threshold: {threshold:.0%}")

    proc = None
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "threshold": threshold,
        "positive_trials": [],
        "negative_events": [],
    }
    try:
        proc, output = start_helper(mode)
        ready, statuses, error = wait_ready(proc, output)
        report["startup_statuses"] = statuses
        if not ready:
            report["startup_error"] = error
            print(f"\nEngine startup: FAIL\nReason: {error}")
            REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1
        print("Engine startup: PASS")
        if args.startup_only:
            REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Report saved to: {REPORT}")
            return 0

        for command, test_phrase in DEFAULT_COMMANDS:
            phrase = f"Laptop {test_phrase}" if wake_required else test_phrase
            print("\n" + "-" * 70)
            print(f"COMMAND: Say '{phrase}' once per attempt.")
            for attempt in range(1, max(1, args.trials) + 1):
                input(f"Attempt {attempt}/{max(1, args.trials)} - press Enter, then speak: ")
                drain(output)
                time.sleep(0.35)
                matched, candidates = collect_one(output, command, max(1.0, args.listen_seconds))
                accepted = bool(
                    matched
                    and matched["kind"] != "DEBUG_REJECTED"
                    and matched["confidence"] >= threshold
                )
                item = {
                    "expected": command,
                    "matched": matched,
                    "accepted_at_configured_threshold": accepted,
                    "candidates": candidates,
                }
                report["positive_trials"].append(item)
                if matched:
                    if matched["kind"] == "DEBUG_REJECTED":
                        result = "REJECTED BY WINDOWS SPEECH"
                    else:
                        result = "PASS" if accepted else "LOW CONFIDENCE"
                    print(
                        f"{result}: heard '{matched['spoken']}' as {matched['command']} "
                        f"at {matched['confidence']:.0%}"
                    )
                else:
                    print("MISS: no matching command was recognized.")

        print("\n" + "-" * 70)
        print("NEGATIVE TEST: Do not say any listed command or wake phrase.")
        print("Normal room speech and background noise are allowed.")
        input("Press Enter to begin: ")
        drain(output)
        time.sleep(0.35)
        negatives = collect_negative(output, max(1.0, args.negative_seconds))
        report["negative_events"] = negatives

        positives = report["positive_trials"]
        passed = sum(item["accepted_at_configured_threshold"] for item in positives)
        false_accepts = sum(
            item["kind"] != "DEBUG_REJECTED" and item["confidence"] >= threshold
            for item in negatives
        )
        total = len(positives)
        gate = total > 0 and passed / total >= 0.90 and false_accepts == 0

        print("\n" + "=" * 70)
        print(f"Command accuracy: {passed}/{total} = {(passed / total * 100) if total else 0:.1f}%")
        print(f"False accepts above threshold: {false_accepts}")
        print(f"Quality gate: {'PASS' if gate else 'NEEDS TUNING'}")
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to: {REPORT}")
        return 0 if gate else 1
    finally:
        stop_helper(proc)


if __name__ == "__main__":
    sys.exit(main())
