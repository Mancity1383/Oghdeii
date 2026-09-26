import os
import shutil
import subprocess
import sys
import webbrowser
from typing import Any

from config_manager import DEFAULT_CONFIG

IS_WINDOWS = sys.platform == "win32"

SAFE_ACTIONS = {
    "copy", "paste", "undo", "redo", "select_all", "screenshot",
    "task_view", "show_desktop", "close_window", "switch_window", "lock",
    "play_pause", "next_track", "prev_track", "mute", "volume_up",
    "volume_down", "open_calculator", "open_notepad", "open_browser",
    "open_terminal",
}

# Platform injection backends, bound only by the branch that supports them
# (user32/SendInput on Windows, pynput elsewhere). mypy walks BOTH sides of a
# computed IS_WINDOWS flag, so the None side of every pairing has to be part of
# the declared type — the platform guard itself remains runtime-only.
user32: Any = None
PynputController: Any = None
PynputKey: Any = None
_linux_keyboard: Any = None

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    VK_BACK = 0x08
    VK_TAB = 0x09
    VK_SHIFT = 0x10
    VK_CONTROL = 0x11
    VK_MENU = 0x12  # Alt
    VK_LWIN = 0x5B
    VK_RWIN = 0x5C
    VK_KEY_C = 0x43
    VK_KEY_V = 0x56
    VK_KEY_S = 0x53
    VK_KEY_Z = 0x5A
    VK_KEY_Y = 0x59
    VK_KEY_A = 0x41
    VK_KEY_D = 0x44
    VK_F4 = 0x73
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF
    VK_MEDIA_NEXT_TRACK = 0xB0
    VK_MEDIA_PREV_TRACK = 0xB1
    VK_MEDIA_STOP = 0xB2
    VK_MEDIA_PLAY_PAUSE = 0xB3

    INPUT_KEYBOARD = 1
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002

    # ULONG_PTR is pointer-sized. Using POINTER(c_ulong) here breaks the INPUT
    # layout on 64-bit Windows and can make SendInput fail intermittently.
    ULONG_PTR = wintypes.WPARAM

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class INPUT_I(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("ii",)
        _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_I)]

    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT

    _EXTENDED_KEYS = {
        VK_LWIN,
        VK_RWIN,
        VK_VOLUME_MUTE,
        VK_VOLUME_DOWN,
        VK_VOLUME_UP,
        VK_MEDIA_NEXT_TRACK,
        VK_MEDIA_PREV_TRACK,
        VK_MEDIA_STOP,
        VK_MEDIA_PLAY_PAUSE,
    }

    def _make_input(vk, is_up=False, is_extended=None):
        if is_extended is None:
            is_extended = vk in _EXTENDED_KEYS
        flags = KEYEVENTF_KEYUP if is_up else 0
        if is_extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        return INPUT(
            type=INPUT_KEYBOARD,
            ii=INPUT_I(ki=KEYBDINPUT(vk, 0, flags, 0, 0)),
        )

    def _send_inputs(inputs):
        if not inputs:
            return
        array_type = INPUT * len(inputs)
        arr = array_type(*inputs)
        sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
        if sent != len(inputs):
            err = ctypes.get_last_error()
            raise OSError(err, f"SendInput sent {sent}/{len(inputs)} events")

    def send_single_key(vk, is_extended=None):
        _send_inputs([
            _make_input(vk, False, is_extended),
            _make_input(vk, True, is_extended),
        ])

    def send_hotkey(*vks):
        events = [_make_input(vk, False) for vk in vks]
        events.extend(_make_input(vk, True) for vk in reversed(vks))
        _send_inputs(events)
else:
    try:
        # Aliased on import: binding the names directly would re-define the
        # module-level declarations above.
        from pynput.keyboard import Controller as _PynputController
        from pynput.keyboard import Key as _PynputKey

        PynputController = _PynputController
        PynputKey = _PynputKey
        _linux_keyboard = PynputController()
    except Exception:
        PynputController = None
        PynputKey = None
        _linux_keyboard = None


def resolve_gesture_action(config, gesture):
    """Single source of truth for knock gesture -> action mapping.

    Used by BOTH the GUI dispatcher and the CLI daemon so they can never
    diverge (previously each had its own copy of this logic with different
    hardcoded fallbacks). Reads the live config on every call, so a mapping
    change takes effect on the very next knock — no restart required.
    """
    actions = config.get("actions", {}) or {}
    count = getattr(gesture, "count", 0)
    side = getattr(gesture, "side", "center")

    # Side-specific double knock (documented feature) overrides the general
    # double-knock mapping only when it is actually configured.
    if count == 2 and side in ("left", "right"):
        side_action = actions.get(f"{side}_double_tap")
        if side_action:
            return side_action

    if count == 1:
        return actions.get("single_tap") or DEFAULT_CONFIG["actions"]["single_tap"]
    if count == 2:
        return actions.get("double_tap") or DEFAULT_CONFIG["actions"]["double_tap"]
    if count >= 3:
        return actions.get("triple_tap") or DEFAULT_CONFIG["actions"]["triple_tap"]
    return None


def resolve_voice_action(config, cmd):
    """Map a recognised voice command word to its configured action."""
    voice_actions = config.get("voice_actions", {}) or {}
    return voice_actions.get(str(cmd).strip().lower())


class ActionExecutor:
    """Cross-platform action dispatcher for knock and voice commands."""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager

    def play_feedback_beep(self):
        try:
            if self.config_manager and not self.config_manager.get("enable_sound_feedback", True):
                return
            if IS_WINDOWS:
                import winsound

                winsound.MessageBeep(winsound.MB_OK)
            elif shutil.which("canberra-gtk-play"):
                subprocess.Popen(
                    ["canberra-gtk-play", "-i", "bell"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif shutil.which("paplay") and os.path.exists("/usr/share/sounds/freedesktop/stereo/bell.oga"):
                subprocess.Popen(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass

    @staticmethod
    def _xdotool_name(key):
        if PynputKey is not None and isinstance(key, PynputKey):
            mapping = {
                PynputKey.ctrl: "ctrl",
                PynputKey.ctrl_l: "ctrl",
                PynputKey.ctrl_r: "ctrl",
                PynputKey.alt: "alt",
                PynputKey.alt_l: "alt",
                PynputKey.alt_r: "alt",
                PynputKey.shift: "shift",
                PynputKey.shift_l: "shift",
                PynputKey.shift_r: "shift",
                PynputKey.cmd: "super",
                PynputKey.cmd_l: "super",
                PynputKey.cmd_r: "super",
                PynputKey.tab: "Tab",
                PynputKey.f4: "F4",
                PynputKey.print_screen: "Print",
            }
            return mapping.get(key, getattr(key, "name", str(key)))
        return str(key)

    def _execute_linux_hotkey(self, *keys):
        """Simulate a hotkey, using pynput first and xdotool as a fallback."""
        if _linux_keyboard is not None:
            pressed = []
            try:
                for key in keys:
                    _linux_keyboard.press(key)
                    pressed.append(key)
                for key in reversed(pressed):
                    _linux_keyboard.release(key)
                return True
            except Exception as exc:
                print(f"[ActionExecutor] pynput hotkey failed: {exc}")
                for key in reversed(pressed):
                    try:
                        _linux_keyboard.release(key)
                    except Exception:
                        pass

        if shutil.which("xdotool"):
            names = [self._xdotool_name(k) for k in keys]
            subprocess.Popen(["xdotool", "key", "--clearmodifiers", "+".join(names)])
            return True

        print("[ActionExecutor] No Linux key injection backend available (pynput/xdotool).")
        return False

    @staticmethod
    def _lock_linux_screen():
        lock_commands = [
            ["loginctl", "lock-session"],
            ["xdg-screensaver", "lock"],
            ["gnome-screensaver-command", "-l"],
            ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver", "Lock"],
            ["slock"],
        ]
        for cmd in lock_commands:
            if shutil.which(cmd[0]):
                try:
                    subprocess.Popen(cmd)
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _linux_volume(action):
        if shutil.which("wpctl"):
            cmd = {
                "mute": ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"],
                "volume_up": ["wpctl", "set-volume", "-l", "1.5", "@DEFAULT_AUDIO_SINK@", "5%+"],
                "volume_down": ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"],
            }[action]
            subprocess.Popen(cmd)
            return True
        if shutil.which("pactl"):
            cmd = {
                "mute": ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                "volume_up": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
                "volume_down": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
            }[action]
            subprocess.Popen(cmd)
            return True
        return False

    def trigger(self, action_name):
        if not action_name or action_name == "none":
            return False
        if action_name not in SAFE_ACTIONS:
            print(f"[ActionExecutor] Rejected unknown or unsafe action: {action_name!r}")
            return False

        print(f"[ActionExecutor] Executing action: {action_name}")
        try:
            if IS_WINDOWS:
                result = self._trigger_windows(action_name)
            else:
                result = self._trigger_linux(action_name)
            if result is False:
                return False
            self.play_feedback_beep()
            return True
        except Exception as exc:
            print(f"[ActionExecutor] Error executing {action_name}: {exc}")
            return False

    def _trigger_windows(self, action_name):
        hotkeys = {
            "copy": (VK_CONTROL, VK_KEY_C),
            "paste": (VK_CONTROL, VK_KEY_V),
            "undo": (VK_CONTROL, VK_KEY_Z),
            "redo": (VK_CONTROL, VK_KEY_Y),
            "select_all": (VK_CONTROL, VK_KEY_A),
            "screenshot": (VK_LWIN, VK_SHIFT, VK_KEY_S),
            "task_view": (VK_LWIN, VK_TAB),
            "show_desktop": (VK_LWIN, VK_KEY_D),
            "close_window": (VK_MENU, VK_F4),
            "switch_window": (VK_MENU, VK_TAB),
        }
        media = {
            "play_pause": VK_MEDIA_PLAY_PAUSE,
            "next_track": VK_MEDIA_NEXT_TRACK,
            "prev_track": VK_MEDIA_PREV_TRACK,
            "mute": VK_VOLUME_MUTE,
            "volume_up": VK_VOLUME_UP,
            "volume_down": VK_VOLUME_DOWN,
        }
        if action_name in hotkeys:
            send_hotkey(*hotkeys[action_name])
        elif action_name in media:
            send_single_key(media[action_name])
        elif action_name == "lock":
            if not user32.LockWorkStation():
                raise ctypes.WinError(ctypes.get_last_error())
        elif action_name == "open_calculator":
            subprocess.Popen(["calc.exe"])
        elif action_name == "open_notepad":
            subprocess.Popen(["notepad.exe"])
        elif action_name == "open_browser":
            if not webbrowser.open("https://www.google.com", new=2):
                raise RuntimeError("The default browser could not be opened")
        elif action_name == "open_terminal":
            subprocess.Popen(["cmd.exe"])
        else:
            raise ValueError(f"Unknown Windows action: {action_name}")
        return True

    def _trigger_linux(self, action_name):
        key = PynputKey
        if action_name in {"copy", "paste", "undo", "redo", "select_all"}:
            char = {"copy": "c", "paste": "v", "undo": "z", "redo": "y", "select_all": "a"}[action_name]
            self._execute_linux_hotkey(key.ctrl if key else "ctrl", char)
        elif action_name == "screenshot":
            if shutil.which("gnome-screenshot"):
                subprocess.Popen(["gnome-screenshot", "-a"])
            elif shutil.which("spectacle"):
                subprocess.Popen(["spectacle", "-r"])
            elif shutil.which("scrot"):
                subprocess.Popen(["scrot", "-s"])
            else:
                self._execute_linux_hotkey(key.print_screen if key else "Print")
        elif action_name == "task_view":
            self._execute_linux_hotkey(key.cmd if key else "super")
        elif action_name == "show_desktop":
            self._execute_linux_hotkey(key.cmd if key else "super", "d")
        elif action_name == "close_window":
            self._execute_linux_hotkey(key.alt if key else "alt", key.f4 if key else "F4")
        elif action_name == "switch_window":
            self._execute_linux_hotkey(key.alt if key else "alt", key.tab if key else "Tab")
        elif action_name in {"play_pause", "next_track", "prev_track"}:
            if shutil.which("playerctl"):
                subcmd = {"play_pause": "play-pause", "next_track": "next", "prev_track": "previous"}[action_name]
                subprocess.Popen(["playerctl", subcmd])
            elif key is not None:
                media_key = {
                    "play_pause": getattr(key, "media_play_pause", None),
                    "next_track": getattr(key, "media_next", None),
                    "prev_track": getattr(key, "media_previous", None),
                }[action_name]
                if media_key is None or not self._execute_linux_hotkey(media_key):
                    raise RuntimeError("No media-control backend available")
            else:
                raise RuntimeError("No media-control backend available")
        elif action_name in {"mute", "volume_up", "volume_down"}:
            if not self._linux_volume(action_name):
                if key is None:
                    raise RuntimeError("No volume-control backend available")
                media_key = {
                    "mute": getattr(key, "media_volume_mute", None),
                    "volume_up": getattr(key, "media_volume_up", None),
                    "volume_down": getattr(key, "media_volume_down", None),
                }[action_name]
                if media_key is None or not self._execute_linux_hotkey(media_key):
                    raise RuntimeError("No volume-control backend available")
        elif action_name == "lock":
            if not self._lock_linux_screen():
                raise RuntimeError("No screen-lock backend available")
        elif action_name == "open_calculator":
            for app in ("gnome-calculator", "kcalc", "xcalc", "galculator"):
                if shutil.which(app):
                    subprocess.Popen([app])
                    break
            else:
                raise RuntimeError("No calculator application found")
        elif action_name == "open_notepad":
            for app in ("gedit", "kate", "mousepad", "xed", "leafpad"):
                if shutil.which(app):
                    subprocess.Popen([app])
                    break
            else:
                raise RuntimeError("No text editor found")
        elif action_name == "open_browser":
            webbrowser.open("https://www.google.com", new=2)
        elif action_name == "open_terminal":
            for app in ("x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal", "xterm"):
                if shutil.which(app):
                    subprocess.Popen([app])
                    break
            else:
                raise RuntimeError("No terminal emulator found")
        else:
            raise ValueError(f"Unknown Linux action: {action_name}")
