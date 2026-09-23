import copy
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_CONFIG = {
    "sensitivity": 0.90,
    "grouping_window_ms": 408,
    "refractory_period_ms": 110,
    "suppress_typing": True,
    "typing_cooldown_ms": 400,
    "detect_sides": True,
    "crowded_mode": False,
    "max_hf_ratio": 0.36,
    "max_rise_time_ms": 4.5,
    "min_crest_factor": 2.05,
    "max_harmonicity": 0.45,
    "side_ratio": 1.35,
    "max_sustained_ms": 55,
    "sustained_revoke_ms": 140,
    "sustained_recovery_ms": 120,
    "voice_like_hold_ms": 900,
    "post_gesture_cooldown_ms": 180,
    "triple_confirm_ms": 220,
    "active_mode": "knock",  # "knock" or "voice"
    "enable_voice_commands": True,
    # Balanced default for softer voices; use Voice Sensitivity for machine-specific tuning.
    "voice_confidence_threshold": 0.55,
    "voice_command_cooldown_ms": 650,
    "voice_require_wake_word": True,
    # v1m System-One cloud intent verification (typesafe-sdk). Off by default:
    # the app stays fully offline until the user opts in and supplies a key.
    "enable_v1m_verification": False,
    "v1m_api_key": "",                # provide locally or via V1M_API_KEY; never ship credentials
    "v1m_endpoint": "https://v1m.ir/v1",
    "v1m_model": "v1m-latest",
    "v1m_wait_timeout_ms": 3000,       # allow time for a typical cloud round trip
    "v1m_request_timeout_ms": 5000,    # HTTP budget for the background call
    "v1m_min_probability": 0.55,       # P(is_valid_command) floor to execute
    "v1m_max_execution_risk": 4.0,     # 0..4 rubric; 4.0 == risk gate off
    "v1m_cache_ttl_s": 600,            # exact-phrase cache lifetime
    "v1m_failure_cooldown_s": 30,      # cloud pause after repeated failures
    "enable_sound_feedback": True,
    "enable_notifications": True,
    "start_with_windows": False,
    "actions": {
        "single_tap": "play_pause",
        "double_tap": "copy",
        "triple_tap": "paste",
        "left_double_tap": "screenshot",
        "right_double_tap": "play_pause",
    },
    "voice_actions": {
        "copy": "copy",
        "paste": "paste",
        "undo": "undo",
        "redo": "redo",
        "select_all": "select_all",
        "screenshot": "screenshot",
        "lock": "lock",
        "show_desktop": "show_desktop",
        "close_window": "close_window",
        "open_calculator": "open_calculator",
        "open_notepad": "open_notepad",
        "open_browser": "open_browser",
        "open_terminal": "open_terminal",
        "play": "play_pause",
        "pause": "play_pause",
        "next": "next_track",
        "previous": "prev_track",
        "mute": "mute",
        "volume_up": "volume_up",
        "volume_down": "volume_down",
    },
    "selected_input_device": None,
    "selected_input_device_signature": None,
}

ALLOWED_ACTIONS = {
    "none", "copy", "paste", "undo", "redo", "select_all", "screenshot",
    "show_desktop", "task_view", "switch_window", "close_window", "lock",
    "play_pause", "next_track", "prev_track", "mute", "volume_up",
    "volume_down", "open_calculator", "open_notepad", "open_browser",
    "open_terminal",
}


class ConfigManager:
    """In-memory config with multi-process safe persistence.

    Several processes read/write config.json (the GUI, calibrate.py,
    voice_calibrate.py, tap_validate.py). Two guarantees matter:

    1. Changes written by another process are picked up automatically
       (throttled stat() check in get() -> reload_if_changed()).
    2. Saving never clobbers keys this process did not change
       (_apply_local_changes performs a per-key merge against the last
       on-disk snapshot instead of dumping a stale full copy).

    Rapid writes (slider drags) are coalesced through a debounce timer so a
    single drag produces one atomic disk write instead of dozens of fsyncs.
    """

    SAVE_DEBOUNCE_S = 0.30          # coalesce rapid slider/checkbox writes
    RELOAD_CHECK_INTERVAL_S = 0.25  # max rate of external-change stat() checks

    def __init__(self, config_dir=None):
        self.config_dir = Path(config_dir) if config_dir is not None else Path.home() / ".laptoptap"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.last_error = None
        self._io_lock = threading.RLock()
        self._listeners = []
        self._dirty = False
        self._debounce_timer = None
        self._next_reload_check = 0.0
        # Snapshot of what we believe is on disk; the diff between this and
        # self.config is exactly "what this process changed".
        self._persisted = copy.deepcopy(DEFAULT_CONFIG)
        self._disk_sig = None
        self.config = self.load()

    @staticmethod
    def _merge_defaults(data):
        merged = copy.deepcopy(DEFAULT_CONFIG)
        if not isinstance(data, dict):
            return merged
        for key, value in data.items():
            if key not in DEFAULT_CONFIG:
                continue
            if key in ("actions", "voice_actions") and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        # The original 200 ms wait cap was shorter than normal cloud latency,
        # so healthy replies opened the circuit after three timeouts. Upgrade
        # that exact legacy default pair on load.
        if (
            merged.get("v1m_wait_timeout_ms") == 200
            and merged.get("v1m_request_timeout_ms") == 2000
        ):
            merged["v1m_wait_timeout_ms"] = DEFAULT_CONFIG["v1m_wait_timeout_ms"]
            merged["v1m_request_timeout_ms"] = DEFAULT_CONFIG["v1m_request_timeout_ms"]
        return ConfigManager._sanitize(merged)

    @staticmethod
    def _sanitize(cfg):
        def as_float(key, default, low, high):
            try:
                value = float(cfg.get(key, default))
            except (TypeError, ValueError):
                value = float(default)
            if value != value or value in (float("inf"), float("-inf")):
                value = float(default)
            cfg[key] = min(high, max(low, value))

        def as_int(key, default, low, high):
            try:
                value = int(cfg.get(key, default))
            except (TypeError, ValueError):
                value = int(default)
            cfg[key] = min(high, max(low, value))

        def as_bool(key, default):
            value = cfg.get(key, default)
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            cfg[key] = bool(value)

        as_float("sensitivity", 0.90, 0.10, 1.0)
        as_float("voice_confidence_threshold", 0.55, 0.20, 0.95)
        as_int("voice_command_cooldown_ms", 650, 0, 10000)
        as_int("v1m_wait_timeout_ms", 3000, 10, 10000)
        as_int("v1m_request_timeout_ms", 5000, 100, 60000)
        as_float("v1m_min_probability", 0.55, 0.0, 1.0)
        as_float("v1m_max_execution_risk", 4.0, 0.0, 4.0)
        as_int("v1m_cache_ttl_s", 600, 0, 86400)
        as_int("v1m_failure_cooldown_s", 30, 0, 3600)
        as_int("grouping_window_ms", 408, 80, 1500)
        as_int("refractory_period_ms", 110, 20, 1000)
        as_int("typing_cooldown_ms", 400, 0, 3000)
        as_float("max_hf_ratio", 0.36, 0.0, 1.0)
        as_float("max_rise_time_ms", 4.5, 0.1, 30.0)
        as_float("min_crest_factor", 2.05, 1.05, 20.0)
        as_float("max_harmonicity", 0.45, 0.05, 0.99)
        as_float("side_ratio", 1.35, 1.01, 5.0)
        as_int("max_sustained_ms", 55, 15, 500)
        as_int("sustained_revoke_ms", 140, 60, 1000)
        as_int("sustained_recovery_ms", 120, 50, 2000)
        as_int("voice_like_hold_ms", 900, 100, 5000)
        as_int("post_gesture_cooldown_ms", 180, 0, 1000)
        as_int("triple_confirm_ms", 220, 140, 1000)
        for key, default in (
            ("suppress_typing", True),
            ("detect_sides", True),
            ("crowded_mode", False),
            ("enable_voice_commands", True),
            ("voice_require_wake_word", True),
            ("enable_v1m_verification", False),
            ("enable_sound_feedback", True),
            ("enable_notifications", True),
            ("start_with_windows", False),
        ):
            as_bool(key, default)
        # Free-form v1m strings: never let garbage into the client constructor.
        cfg["v1m_endpoint"] = str(cfg.get("v1m_endpoint") or DEFAULT_CONFIG["v1m_endpoint"]).strip()[:200]
        cfg["v1m_model"] = str(cfg.get("v1m_model") or DEFAULT_CONFIG["v1m_model"]).strip()[:64]
        cfg["v1m_api_key"] = str(cfg.get("v1m_api_key") or "").strip()[:256]
        if cfg.get("crowded_mode"):
            cfg["detect_sides"] = True
        if cfg.get("active_mode") not in ("knock", "voice"):
            cfg["active_mode"] = "knock"
        try:
            selected = cfg.get("selected_input_device")
            cfg["selected_input_device"] = None if selected is None else max(0, int(selected))
        except (TypeError, ValueError):
            cfg["selected_input_device"] = None
        signature = cfg.get("selected_input_device_signature")
        cfg["selected_input_device_signature"] = str(signature)[:500] if signature else None
        for group in ("actions", "voice_actions"):
            defaults = DEFAULT_CONFIG[group]
            values = cfg.get(group) if isinstance(cfg.get(group), dict) else {}
            cfg[group] = {
                key: values.get(key) if values.get(key) in ALLOWED_ACTIONS else default
                for key, default in defaults.items()
            }
        return cfg

    def load(self):
        """Load config.json (or defaults) and adopt it as our disk snapshot."""
        config = None
        if self.config_file.exists():
            try:
                with self.config_file.open("r", encoding="utf-8") as f:
                    config = self._merge_defaults(json.load(f))
            except Exception as exc:
                print(f"[Config] Error loading config: {exc}. Using defaults.")
        if config is None:
            config = copy.deepcopy(DEFAULT_CONFIG)
        self.config = config
        with self._io_lock:
            self._persisted = copy.deepcopy(config)
            self._dirty = False
            self._disk_sig = self._disk_signature()
        return config

    # ------------------------------------------------------------------
    # Multi-process synchronisation
    # ------------------------------------------------------------------

    def _disk_signature(self):
        try:
            st = self.config_file.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _apply_local_changes(self, disk_config):
        """Overlay this process's edits onto the latest on-disk state.

        Prevents the classic lost-update bug: calibrate.py (a separate
        process) may have rewritten config.json since we last read it, so
        dumping our full in-memory copy would silently revert its results —
        and vice versa. We only overwrite keys this process actually changed.
        """
        missing = object()
        out = copy.deepcopy(disk_config)
        base = self._persisted
        for key, value in self.config.items():
            previous = base.get(key, missing)
            if previous is not missing and value == previous:
                continue  # unchanged here -> keep whatever the other process wrote
            if (
                key in ("actions", "voice_actions")
                and isinstance(value, dict)
                and isinstance(previous, dict)
            ):
                group = dict(out.get(key) or {})
                for sub_key, sub_value in value.items():
                    if sub_value != previous.get(sub_key, missing):
                        group[sub_key] = sub_value
                out[key] = group
            else:
                out[key] = copy.deepcopy(value)
        return out

    def reload_if_changed(self, force=False):
        """Adopt config.json changes written by another process.

        Throttled to RELOAD_CHECK_INTERVAL_S so the hot audio path can call
        it through get() without paying a stat() on every lookup.
        """
        now = time.monotonic()
        if not force and now < self._next_reload_check:
            return False
        self._next_reload_check = now + self.RELOAD_CHECK_INTERVAL_S
        sig = self._disk_signature()
        if sig is None or sig == self._disk_sig:
            return False
        with self._io_lock:
            sig = self._disk_signature()
            if sig is None or sig == self._disk_sig:
                return False
            try:
                with self.config_file.open("r", encoding="utf-8") as f:
                    disk_config = self._merge_defaults(json.load(f))
            except Exception:
                # Atomic-replace race or corrupt file: retry soon, adopt nothing.
                self._next_reload_check = time.monotonic() + 0.05
                return False
            self._disk_sig = sig
            if self._dirty:
                # We have unsaved local edits: merge theirs under ours and stay
                # dirty so the pending save persists the merged result.
                merged = self._sanitize(self._apply_local_changes(disk_config))
                changed = merged != self.config
                self.config = merged
            else:
                changed = disk_config != self.config
                self.config = disk_config
                self._persisted = copy.deepcopy(disk_config)
        if changed:
            self._notify()
            return True
        return False

    def add_listener(self, callback):
        """Subscribe to config changes (own writes and external reloads).

        May be invoked from a non-GUI thread (audio worker detecting an
        external edit); Qt users should marshal to the GUI thread.
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self):
        for listener in list(self._listeners):
            try:
                listener()
            except Exception as exc:
                print(f"[Config] Listener error: {exc}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _cancel_debounce_locked(self):
        timer = self._debounce_timer
        self._debounce_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _schedule_save(self):
        with self._io_lock:
            if self._debounce_timer is not None:
                return  # a save is already pending; it will pick up this change
            timer = threading.Timer(self.SAVE_DEBOUNCE_S, self._debounced_save)
            timer.daemon = True
            self._debounce_timer = timer
            timer.start()

    def _debounced_save(self):
        with self._io_lock:
            self._debounce_timer = None
            if not self._dirty:
                return
        try:
            self.save()
        except Exception as exc:
            print(f"[Config] Debounced save failed: {exc}")

    def flush(self):
        """Force any debounced change onto disk immediately (call on exit)."""
        with self._io_lock:
            dirty = self._dirty
            self._cancel_debounce_locked()
        if dirty:
            return self.save()
        return True

    def save(self):
        """Atomically persist config, merging in changes made by other processes.

        A crash cannot leave a half-written JSON file, and a stale copy from
        this process can no longer clobber keys another process just wrote.
        """
        with self._io_lock:
            self._cancel_debounce_locked()
            disk_config = None
            if self.config_file.exists():
                try:
                    with self.config_file.open("r", encoding="utf-8") as f:
                        disk_config = self._merge_defaults(json.load(f))
                except Exception:
                    disk_config = None  # unreadable -> fall back to our copy
            before = self.config
            if disk_config is None:
                merged = self._sanitize(self.config)
            else:
                merged = self._sanitize(self._apply_local_changes(disk_config))
            adopted_external = merged != before
            self.config = merged
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self.config_dir,
                    prefix="config.", suffix=".tmp", delete=False
                ) as f:
                    temp_name = f.name
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_name, self.config_file)
                self.last_error = None
                self._persisted = copy.deepcopy(self.config)
                self._dirty = False
                self._disk_sig = self._disk_signature()
                result = True
            except Exception as exc:
                self.last_error = str(exc)
                self._dirty = True  # keep edits pending so a later flush retries
                print(f"[Config] Error saving config: {exc}")
                if temp_name:
                    try:
                        Path(temp_name).unlink(missing_ok=True)
                    except Exception:
                        pass
                result = False
        if adopted_external:
            self._notify()
        return result

    def get(self, key, default=None):
        self.reload_if_changed()
        return self.config.get(key, default)

    def set(self, key, value, flush=True):
        """Update a key in memory (visible to all consumers immediately).

        flush=True (default) writes to disk synchronously. flush=False defers
        the disk write through SAVE_DEBOUNCE_S — use it for high-frequency
        updates such as slider drags; call flush() before exiting.
        """
        self.reload_if_changed()
        previous = self.config.get(key)
        self.config[key] = value
        self._dirty = True
        autostart_key = key in ("start_with_windows", "start_with_system")
        if flush or autostart_key:
            saved = self.save()
        else:
            self._schedule_save()
            saved = True
        if autostart_key:
            if os.name == "nt":
                registry_ok = self.set_autostart_registry(bool(value))
                if not registry_ok:
                    registry_error = self.last_error
                    self.config[key] = previous
                    self.save()
                    self.last_error = registry_error
                saved = registry_ok and saved
            else:
                self.set_autostart_linux(bool(value))
        self._notify()
        return saved

    def update(self, values, flush=True):
        if not isinstance(values, dict):
            raise TypeError("Config update must be a dictionary")
        self.reload_if_changed()
        self.config.update(values)
        self._dirty = True
        if flush:
            saved = self.save()
        else:
            self._schedule_save()
            saved = True
        self._notify()
        return saved

    def set_autostart_registry(self, enable: bool):
        if os.name != "nt":
            return False
        try:
            import winreg

            run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as key:
                app_name = "LaptopTap"
                if enable:
                    if getattr(sys, "frozen", False):
                        cmd = f'"{Path(sys.executable).resolve()}" --minimized'
                    else:
                        main_py = (Path(__file__).parent / "main.py").resolve()
                        pythonw = Path(sys.executable).parent / "pythonw.exe"
                        if not pythonw.exists():
                            pythonw = Path(sys.executable)
                        cmd = f'"{pythonw}" "{main_py}" --minimized'
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, cmd)
                    print(f"[Config] Windows Autostart registered: {cmd}")
                else:
                    try:
                        winreg.DeleteValue(key, app_name)
                        print("[Config] Windows Autostart removed.")
                    except FileNotFoundError:
                        pass
            return True
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[Config] Error updating autostart registry: {exc}")
            return False

    def set_autostart_linux(self, enable: bool):
        try:
            autostart_dir = Path.home() / ".config" / "autostart"
            autostart_dir.mkdir(parents=True, exist_ok=True)
            desktop_file = autostart_dir / "laptoptap.desktop"
            if enable:
                main_py = (Path(__file__).parent / "main.py").resolve()
                venv_py = (Path(__file__).parent / ".venv" / "bin" / "python3").resolve()
                python_exec = venv_py if venv_py.exists() else Path(sys.executable).resolve()
                icon_path = (Path(__file__).parent / "assets" / "icon.png").resolve()
                # Desktop-entry Exec fields use double-quote escaping, not shell quoting.
                def desktop_quote(value):
                    value = str(value).replace("\\", "\\\\").replace('"', '\\"')
                    return f'"{value}"'
                exec_line = f"{desktop_quote(python_exec)} {desktop_quote(main_py)} --minimized"
                content = f"""[Desktop Entry]
Type=Application
Name=عقده ای
Comment=میانبرهای ضربه به بدنه و فرمان صوتی
Exec={exec_line}
Icon={icon_path}
Terminal=false
Categories=Utility;Audio;
X-GNOME-Autostart-enabled=true
"""
                desktop_file.write_text(content, encoding="utf-8")
                print(f"[Config] Linux autostart file written: {desktop_file}")
            elif desktop_file.exists():
                desktop_file.unlink()
                print("[Config] Linux autostart file removed.")
        except Exception as exc:
            print(f"[Config] Error updating Linux autostart: {exc}")
