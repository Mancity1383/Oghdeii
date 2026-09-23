import os
import sys
import threading
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIcon, QAction, QPainterPath
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QCheckBox, QPushButton, QFrame,
    QSystemTrayIcon, QMenu, QStackedWidget, QGridLayout, QScrollArea, QSizePolicy,
    QLineEdit, QDialog, QPlainTextEdit,
)

from config_manager import ConfigManager, DEFAULT_CONFIG
from audio_engine import AudioEngine
from tap_detector import TapDetector, DetectedTapGesture
from action_executor import ActionExecutor, resolve_gesture_action, resolve_voice_action

ACTION_LABELS = {
    "copy": "📋 Copy (Ctrl + C)",
    "paste": "📄 Paste (Ctrl + V)",
    "undo": "↩ Undo (Ctrl + Z)",
    "redo": "↪ Redo (Ctrl + Y)",
    "select_all": "📑 Select All (Ctrl + A)",
    "screenshot": "📸 Screenshot (Win + Shift + S)",
    "show_desktop": "🖥 Show / Hide Desktop (Win + D)",
    "task_view": "🗂 Task View (Win + Tab)",
    "switch_window": "🔄 Switch Window (Alt + Tab)",
    "close_window": "❌ Close Window (Alt + F4)",
    "lock": "🔒 Lock PC (Win + L)",
    "play_pause": "🎵 Play / Pause Media",
    "next_track": "⏭ Next Track",
    "prev_track": "⏮ Previous Track",
    "mute": "🔇 Toggle Mute",
    "volume_up": "🔊 Volume Up",
    "volume_down": "🔉 Volume Down",
    "open_calculator": "🔢 Open Calculator",
    "open_notepad": "📝 Open Notepad",
    "open_browser": "🌐 Open Browser",
    "open_terminal": "💻 Open Terminal",
    "none": "🚫 Do Nothing"
}

# Example phrases shown beside editable V1M action mappings. Free-form speech
# is sent to V1M; these examples are not a local recognizer allowlist.
VOICE_COMMAND_ROWS = [
    ("Copy / Copy That", ("copy",)),
    ("Paste / Paste That", ("paste",)),
    ("Undo / Undo That", ("undo",)),
    ("Redo", ("redo",)),
    ("Select All", ("select_all",)),
    ("Screenshot / Take Screenshot", ("screenshot",)),
    ("Desktop / Show Desktop", ("show_desktop",)),
    ("Close Window / Close App", ("close_window",)),
    ("Lock PC / Lock Screen", ("lock",)),
    ("Play / Pause Music", ("play", "pause")),
    ("Next / Next Track", ("next",)),
    ("Previous / Previous Track", ("previous",)),
    ("Mute / Mute Sound", ("mute",)),
    ("Volume Up / Turn Up", ("volume_up",)),
    ("Volume Down / Turn Down", ("volume_down",)),
    ("Calculator", ("open_calculator",)),
    ("Notepad", ("open_notepad",)),
    ("Browser", ("open_browser",)),
    ("Terminal", ("open_terminal",)),
]


def make_card(title=None, title_color="#00f0ff"):
    """Creates a modern dark frameless card container avoiding Windows QGroupBox white artifacts."""
    frame = QFrame()
    frame.setStyleSheet("""
        QFrame {
            background-color: #121520;
            border: 1px solid #1f2537;
            border-radius: 8px;
        }
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 9, 10, 9)
    layout.setSpacing(6)
    if title:
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {title_color}; font-weight: 700; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(lbl)
    return frame, layout


class WaveformWidget(QWidget):
    """Compact oscilloscope showing chassis vibration, trigger threshold, and VU level bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(75)
        self.setMaximumHeight(85)
        self.history = np.zeros(512, dtype=np.float32)
        self.current_peak = 0.0
        self.threshold = 0.05
        self.flash_alpha = 0.0

    def update_data(self, history, current_peak, threshold):
        if len(history) >= 512:
            self.history = history[-512:]
        else:
            self.history = history
        self.current_peak = current_peak
        self.threshold = threshold
        if self.flash_alpha > 0:
            self.flash_alpha = max(0.0, self.flash_alpha - 0.08)
        self.update()

    def flash(self):
        self.flash_alpha = 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        vu_w = 10
        vu_margin = 6
        wave_w = w - vu_w - vu_margin * 2
        mid_y = h / 2.0

        # Deep dark background
        painter.fillRect(0, 0, w, h, QColor(10, 12, 18))

        # Grid line
        grid_pen = QPen(QColor(28, 34, 48), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        painter.drawLine(0, int(mid_y), int(wave_w), int(mid_y))

        # Flash overlay on tap
        if self.flash_alpha > 0:
            flash_color = QColor(0, 240, 255, int(self.flash_alpha * 50))
            painter.fillRect(0, 0, int(wave_w), h, flash_color)

        # Threshold lines (Top & Bottom)
        scale = (mid_y - 8)
        thresh_y_top = mid_y - min(scale, self.threshold * scale * 3.5)
        thresh_y_bot = mid_y + min(scale, self.threshold * scale * 3.5)

        thresh_pen = QPen(QColor(255, 171, 0, 180), 1, Qt.PenStyle.DotLine)
        painter.setPen(thresh_pen)
        painter.drawLine(0, int(thresh_y_top), int(wave_w), int(thresh_y_top))
        painter.drawLine(0, int(thresh_y_bot), int(wave_w), int(thresh_y_bot))

        # Waveform line: min/max envelope per pixel column, drawn as ONE path
        # (single stroke). The previous version issued up to 512 individual
        # drawLine calls per frame in a Python loop — the main UI hotspot.
        if len(self.history) > 1:
            wave_color = QColor(0, 240, 255) if self.current_peak < self.threshold else QColor(255, 64, 129)
            wave_pen = QPen(wave_color, 2)
            painter.setPen(wave_pen)

            hist = np.asarray(self.history, dtype=np.float32)
            n = hist.size
            cols = max(2, min(n, int(wave_w)))
            edges = np.linspace(0, n, cols + 1).astype(np.int64)
            starts = edges[:-1]
            counts = np.diff(edges)
            valid = counts > 0
            starts = starts[valid]
            if starts.size:
                lows = np.minimum.reduceat(hist, starts)
                highs = np.maximum.reduceat(hist, starts)
                dx = wave_w / float(max(1, cols - 1))
                path = QPainterPath()
                for i in range(lows.size):
                    x = i * dx
                    y_hi = mid_y - float(highs[i]) * scale * 3.5
                    y_lo = mid_y - float(lows[i]) * scale * 3.5
                    y_hi = max(3.0, min(h - 3.0, y_hi))
                    y_lo = max(3.0, min(h - 3.0, y_lo))
                    if i == 0:
                        path.moveTo(x, y_hi)
                    else:
                        path.lineTo(x, y_hi)
                    path.lineTo(x, y_lo)
                    path.lineTo(x, y_hi)  # back to top before the next column
                painter.drawPath(path)

        # VU Level Meter on the right
        vu_x = wave_w + vu_margin
        vu_h = h - 12
        vu_y = 6
        painter.fillRect(int(vu_x), int(vu_y), int(vu_w), int(vu_h), QColor(20, 24, 34))

        norm_peak = min(1.0, max(0.0, self.current_peak * 3.5))
        level_h = int(norm_peak * vu_h)
        if level_h > 0:
            bar_color = QColor(0, 230, 118) if self.current_peak < self.threshold else QColor(255, 23, 68)
            painter.fillRect(int(vu_x), int(vu_y + vu_h - level_h), int(vu_w), level_h, bar_color)

        painter.setPen(QPen(QColor(38, 46, 64), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(int(vu_x), int(vu_y), int(vu_w), int(vu_h))

        # Outer border
        painter.setPen(QPen(QColor(30, 36, 50), 1))
        painter.drawRect(0, 0, w - 1, h - 1)


class SignalBridge(QObject):
    gesture_signal = pyqtSignal(object)
    level_signal = pyqtSignal(float, float)
    # (command to resolve, recogniser confidence, spoken text, v1m decision|None)
    voice_signal = pyqtSignal(str, float, str, object)
    # Emitted when the v1m guardrail vetoed a phrase (never executes).
    voice_reject_signal = pyqtSignal(object)
    voice_status_signal = pyqtSignal(str, str)
    # (success, message) from the background connection test.
    v1m_test_signal = pyqtSignal(bool, str)
    config_signal = pyqtSignal()


class MainWindow(QMainWindow):
    def __init__(self, config_manager, audio_engine, tap_detector, action_executor,
                 voice_detector=None, voice_verifier=None):
        super().__init__()
        self.config = config_manager
        self.audio_engine = audio_engine
        self.tap_detector = tap_detector
        self.executor = action_executor
        self.voice_detector = voice_detector
        self.voice_verifier = voice_verifier

        self.bridge = SignalBridge()
        self.bridge.gesture_signal.connect(self.on_gesture_received)
        self.bridge.level_signal.connect(self.on_level_received)
        self.bridge.voice_signal.connect(self.on_voice_received)
        self.bridge.voice_reject_signal.connect(self.on_voice_rejected)
        self.bridge.voice_status_signal.connect(self.on_voice_status_changed)
        self.bridge.v1m_test_signal.connect(self.on_v1m_test_result)
        self.bridge.config_signal.connect(self._refresh_from_config)

        # React to config changes made by *any* writer: our own widgets and
        # external processes (calibrate.py / voice_calibrate.py rewrite
        # config.json). The listener may fire on the audio worker thread, so it
        # only emits a Qt signal; the slot runs on the GUI thread.
        self.config.add_listener(lambda: self.bridge.config_signal.emit())

        self.tap_detector.on_gesture_detected = lambda g: self.bridge.gesture_signal.emit(g)
        self.tap_detector.on_level_update = lambda p, t: self.bridge.level_signal.emit(p, t)
        if self.voice_detector:
            # The v1m gate runs HERE, on the recogniser's reader thread, before
            # the Qt hand-off: it is bounded by v1m_wait_timeout_ms (3000 ms) and
            # never raises, so the helper's stdout pipe keeps draining.
            self.voice_detector.on_voice_command = self._on_voice_captured
            self.voice_detector.on_status_change = lambda state, msg="": self.bridge.voice_status_signal.emit(state, msg)

        self.current_peak = 0.0
        self.current_threshold = 0.05
        self.is_monitoring_paused = False

        self.init_ui()
        self.init_tray()

        initial_mode = self.config.get("active_mode", "knock")
        self.switch_mode(initial_mode)

        # Display-only timer: started by showEvent(), stopped by hideEvent().
        # This way a tray-minimized instance burns ~no CPU refreshing a
        # waveform nobody can see.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_display)

    def init_ui(self):
        self.setWindowTitle("عقده ای — میانبرهای ضربه و صدا")
        from pathlib import Path
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(640, 720)
        self.setMinimumSize(500, 560)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0d13;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI Variable Display', 'Segoe UI', sans-serif;
            }
            QComboBox {
                background-color: #171b26;
                border: 1px solid #293044;
                border-radius: 5px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QComboBox:hover {
                border: 1px solid #00f0ff;
                background-color: #1c2230;
            }
            QComboBox QAbstractItemView {
                background-color: #121520;
                border: 1px solid #293044;
                selection-background-color: #00f0ff;
                selection-color: #0b0d13;
                color: #ffffff;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 5px;
                background: #1c2230;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #00f0ff);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #00f0ff;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QCheckBox {
                spacing: 6px;
                font-size: 11px;
                color: #cbd5e1;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border-radius: 3px;
                border: 1px solid #2f384c;
                background-color: #141722;
            }
            QCheckBox::indicator:checked {
                background-color: #00f0ff;
                border-color: #00f0ff;
            }
            QPushButton {
                background-color: #151824;
                border: 1px solid #262d3e;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 600;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #1e2434;
                border: 1px solid #00f0ff;
            }
            QPushButton:pressed {
                background-color: #0d0f16;
            }
        """)

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(7)

        # 1. Header Bar (Fixed)
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_label = QLabel("⚡ عقده ای")
        title_label.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff;")
        title_sub = QLabel("میانبرهای ضربه به بدنه و فرمان صوتی")
        title_sub.setStyleSheet("font-size: 10px; color: #64748b; font-weight: 500;")
        title_box.addWidget(title_label)
        title_box.addWidget(title_sub)
        header_layout.addLayout(title_box)

        self.status_badge = QLabel("● Knock Mode Active")
        self.status_badge.setStyleSheet("font-size: 10px; font-weight: bold; color: #00f0ff; background-color: #0c2b32; padding: 4px 10px; border-radius: 10px; border: 1px solid #00f0ff;")
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        main_layout.addLayout(header_layout)

        # 2. Segmented Mode Switcher (Fixed)
        mode_container = QFrame()
        mode_container.setStyleSheet("""
            QFrame {
                background-color: #121520;
                border: 1px solid #1f2537;
                border-radius: 8px;
                padding: 2px;
            }
        """)
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(2, 2, 2, 2)
        mode_layout.setSpacing(4)

        self.btn_mode_knock = QPushButton("⚡ Chassis Knock Mode")
        self.btn_mode_knock.setCheckable(True)
        self.btn_mode_knock.setChecked(True)
        self.btn_mode_knock.clicked.connect(lambda: self.switch_mode("knock"))

        self.btn_mode_voice = QPushButton("🎙️ Voice Control Mode")
        self.btn_mode_voice.setCheckable(True)
        self.btn_mode_voice.setChecked(False)
        self.btn_mode_voice.clicked.connect(lambda: self.switch_mode("voice"))

        mode_layout.addWidget(self.btn_mode_knock, 1)
        mode_layout.addWidget(self.btn_mode_voice, 1)
        main_layout.addWidget(mode_container)

        # 3. Dynamic Event Trigger Banner (Fixed)
        self.event_badge = QLabel("Ready. Knock on chassis or switch to Voice Mode.")
        self.event_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.event_badge.setWordWrap(True)
        self.event_badge.setStyleSheet("""
            background-color: #121522;
            border: 1px dashed #242c3e;
            border-radius: 6px;
            padding: 6px;
            font-size: 11px;
            font-weight: 600;
            color: #94a3b8;
        """)
        main_layout.addWidget(self.event_badge)

        # 4. Scrollable Middle Area (With NO horizontal scrollbar and sleek dark vertical bar)
        scroll_area = QScrollArea()
        self.scroll_area = scroll_area
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #0f121a;
                width: 6px;
                border-radius: 3px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #2b3348;
                border-radius: 3px;
                min-height: 25px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00f0ff;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 2, 4, 2)
        scroll_content_layout.setSpacing(7)

        # Stacked Widget (Knock Panel vs Voice Panel)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background: transparent;")

        # ====== PANEL 1: KNOCK MODE ======
        self.panel_knock = QWidget()
        self.panel_knock.setStyleSheet("background: transparent;")
        knock_layout = QVBoxLayout(self.panel_knock)
        knock_layout.setContentsMargins(0, 0, 0, 0)
        knock_layout.setSpacing(7)

        # Card 1: Oscilloscope
        card_osc, osc_vbox = make_card("Live Mechanical Vibration (Real-Time DSP)", "#00f0ff")
        self.waveform_widget = WaveformWidget(self)
        osc_vbox.addWidget(self.waveform_widget)
        knock_layout.addWidget(card_osc)

        # Card 2: Hardware & Sensitivity
        card_hw, hw_layout = make_card("Hardware & Shock Sensitivity", "#00f0ff")

        mic_row = QHBoxLayout()
        mic_lbl = QLabel("Microphone:")
        mic_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; background: transparent;")
        self.combo_mic = QComboBox()
        self._populate_audio_devices()
        self.combo_mic.currentIndexChanged.connect(self.on_device_changed)
        mic_row.addWidget(mic_lbl)
        mic_row.addWidget(self.combo_mic, 1)
        hw_layout.addLayout(mic_row)

        sens_row = QHBoxLayout()
        sens_label = QLabel("Shock Sensitivity:")
        sens_label.setStyleSheet("color: #e2e8f0; font-size: 11px; background: transparent;")
        self.sens_val_label = QLabel(f"{int(self.config.get('sensitivity', 0.90) * 100)}%")
        self.sens_val_label.setStyleSheet("color: #00f0ff; font-weight: bold; font-size: 11px; background: transparent;")
        sens_row.addWidget(sens_label)
        sens_row.addStretch()
        sens_row.addWidget(self.sens_val_label)
        hw_layout.addLayout(sens_row)

        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(10, 100)
        self.sens_slider.setValue(int(self.config.get("sensitivity", 0.90) * 100))
        self.sens_slider.valueChanged.connect(self.on_sensitivity_changed)
        hw_layout.addWidget(self.sens_slider)
        knock_layout.addWidget(card_hw)

        # Card 3: Tap Shortcuts
        card_actions, actions_grid = make_card("Chassis Knock Shortcuts", "#00f0ff")
        grid = QGridLayout()
        grid.setSpacing(6)

        lbl_1 = QLabel("Single Knock (1×):")
        lbl_1.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        self.combo_1 = QComboBox()
        self._populate_actions(self.combo_1, self.config.get("actions", {}).get("single_tap"))
        self.combo_1.currentTextChanged.connect(lambda txt: self._save_action("single_tap", self.combo_1))
        grid.addWidget(lbl_1, 0, 0)
        grid.addWidget(self.combo_1, 0, 1)

        lbl_2 = QLabel("Double Knock (2×):")
        lbl_2.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        self.combo_2 = QComboBox()
        self._populate_actions(self.combo_2, self.config.get("actions", {}).get("double_tap"))
        self.combo_2.currentTextChanged.connect(lambda txt: self._save_action("double_tap", self.combo_2))
        grid.addWidget(lbl_2, 1, 0)
        grid.addWidget(self.combo_2, 1, 1)

        lbl_3 = QLabel("Triple Knock (3×):")
        lbl_3.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        self.combo_3 = QComboBox()
        self._populate_actions(self.combo_3, self.config.get("actions", {}).get("triple_tap"))
        self.combo_3.currentTextChanged.connect(lambda txt: self._save_action("triple_tap", self.combo_3))
        grid.addWidget(lbl_3, 2, 0)
        grid.addWidget(self.combo_3, 2, 1)

        # Side-specific double-knock mappings used to be hidden, hardcoded
        # defaults (left→screenshot, right→play_pause) that silently overrode
        # whatever the user picked in "Double Knock". Expose them so the
        # override is visible and editable.
        lbl_l = QLabel("Left-Side Double (2×):")
        lbl_l.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        self.combo_left = QComboBox()
        self._populate_actions(self.combo_left, self.config.get("actions", {}).get("left_double_tap"))
        self.combo_left.currentTextChanged.connect(lambda txt: self._save_action("left_double_tap", self.combo_left))
        grid.addWidget(lbl_l, 3, 0)
        grid.addWidget(self.combo_left, 3, 1)

        lbl_r = QLabel("Right-Side Double (2×):")
        lbl_r.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        self.combo_right = QComboBox()
        self._populate_actions(self.combo_right, self.config.get("actions", {}).get("right_double_tap"))
        self.combo_right.currentTextChanged.connect(lambda txt: self._save_action("right_double_tap", self.combo_right))
        grid.addWidget(lbl_r, 4, 0)
        grid.addWidget(self.combo_right, 4, 1)

        side_hint = QLabel(
            "Side rows apply only when the knock is localized to that chassis "
            "edge; otherwise “Double Knock” is used."
        )
        side_hint.setWordWrap(True)
        side_hint.setStyleSheet("color: #64748b; font-size: 10px; background: transparent;")
        grid.addWidget(side_hint, 5, 0, 1, 2)

        self._action_combo_map = [
            (self.combo_1, "single_tap"),
            (self.combo_2, "double_tap"),
            (self.combo_3, "triple_tap"),
            (self.combo_left, "left_double_tap"),
            (self.combo_right, "right_double_tap"),
        ]

        actions_grid.addLayout(grid)
        knock_layout.addWidget(card_actions)

        self.stacked_widget.addWidget(self.panel_knock)

        # ====== PANEL 2: VOICE MODE ======
        self.panel_voice = QWidget()
        self.panel_voice.setStyleSheet("background: transparent;")
        voice_layout = QVBoxLayout(self.panel_voice)
        voice_layout.setContentsMargins(0, 0, 0, 0)
        voice_layout.setSpacing(7)

        card_voice, vs_vbox = make_card("Voice Commands & V1M Action Selection", "#c084fc")

        self.voice_desc = QLabel()
        self.voice_desc.setStyleSheet("font-size: 11px; color: #cbd5e1; background: transparent;")
        self.voice_desc.setWordWrap(True)
        vs_vbox.addWidget(self.voice_desc)

        # Offline confidence floor; low-confidence speech is passed to V1M when
        # cloud classification is enabled, so V1M can use the N-best context.
        v_sens_row = QHBoxLayout()
        v_sens_lbl = QLabel("Minimum Speech Recognition Confidence:")
        v_sens_lbl.setStyleSheet("color: #e2e8f0; font-size: 11px; background: transparent;")
        conf_val = self.config.get("voice_confidence_threshold", 0.55)
        self.voice_sens_val_label = QLabel(f"{int(conf_val * 100)}%")
        self.voice_sens_val_label.setStyleSheet("color: #c084fc; font-weight: bold; font-size: 11px; background: transparent;")
        v_sens_row.addWidget(v_sens_lbl)
        v_sens_row.addStretch()
        v_sens_row.addWidget(self.voice_sens_val_label)
        vs_vbox.addLayout(v_sens_row)

        self.voice_sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.voice_sens_slider.setRange(30, 95)
        self.voice_sens_slider.setToolTip(
            "Lower values let quieter or uncertain phrases through in offline mode, "
            "with more chance of a false match. With V1M enabled, low-confidence "
            "recognition candidates are sent to V1M for intent classification."
        )
        self.voice_sens_slider.setValue(int(conf_val * 100))
        self.voice_sens_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e2436;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #818cf8, stop:1 #c084fc);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #c084fc;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        self.voice_sens_slider.valueChanged.connect(self.on_voice_sensitivity_changed)
        # No sliderReleased restart: VoiceDetector watches the config and
        # respawns the helper once (debounced) after the drag settles.
        vs_vbox.addWidget(self.voice_sens_slider)

        voice_v1m_hint = QLabel(
            "Windows listens with a constrained command vocabulary for better accuracy. "
            "The recognized phrase and alternate readings are sent to V1M, which decides "
            "whether to run an action and which configured action it means."
        )
        voice_v1m_hint.setWordWrap(True)
        voice_v1m_hint.setStyleSheet("color: #94a3b8; font-size: 10px; background: transparent;")
        vs_vbox.addWidget(voice_v1m_hint)

        # Scroll area for voice commands list
        v_scroll = QScrollArea()
        v_scroll.setWidgetResizable(True)
        v_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        v_scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #1e2436; border-radius: 6px; background-color: #0d0f16; }
            QScrollBar:vertical { background: #10121a; width: 5px; border-radius: 2px; }
            QScrollBar::handle:vertical { background: #3b4562; border-radius: 2px; }
        """)
        v_scroll_content = QWidget()
        v_scroll_content.setStyleSheet("background: transparent;")
        v_scroll_layout = QVBoxLayout(v_scroll_content)
        v_scroll_layout.setContentsMargins(4, 4, 4, 4)
        v_scroll_layout.setSpacing(4)

        self.voice_phrase_labels = []
        self._voice_combo_map = []
        voice_actions_cfg = self.config.get("voice_actions", {}) or {}
        for phrase, cmd_keys in VOICE_COMMAND_ROWS:
            row = QGridLayout()
            row.setColumnStretch(0, 1)
            row.setColumnStretch(1, 1)
            lbl_phrase = QLabel(f"🗣️ \"{phrase}\"")
            lbl_phrase.setStyleSheet("font-weight: 700; color: #c084fc; font-size: 10px; background: transparent;")
            lbl_phrase.setWordWrap(True)
            lbl_phrase.setMinimumWidth(0)
            lbl_phrase.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.voice_phrase_labels.append((lbl_phrase, phrase))
            # Editable binding: which action this spoken phrase triggers.
            # Saved live through _save_voice_action — no restart needed.
            combo_action = QComboBox()
            self._populate_actions(combo_action, voice_actions_cfg.get(cmd_keys[0], "none"))
            combo_action.currentIndexChanged.connect(
                lambda _i, keys=cmd_keys, c=combo_action: self._save_voice_action(keys, c)
            )
            combo_action.setMinimumWidth(0)
            combo_action.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            combo_action.setToolTip(
                "Action triggered by this voice phrase (changes apply to the next utterance)."
            )
            self._voice_combo_map.append((combo_action, cmd_keys))
            row.addWidget(lbl_phrase, 0, 0)
            row.addWidget(combo_action, 0, 1)

            line_frame = QFrame()
            line_frame.setStyleSheet("background-color: #141722; border-radius: 4px; padding: 2px 6px; border: none;")
            line_frame.setLayout(row)
            v_scroll_layout.addWidget(line_frame)

        v_scroll.setWidget(v_scroll_content)
        v_scroll.setFixedHeight(130)
        vs_vbox.addWidget(v_scroll)
        voice_layout.addWidget(card_voice)
        self._refresh_voice_instructions()

        self.stacked_widget.addWidget(self.panel_voice)
        scroll_content_layout.addWidget(self.stacked_widget)

        # Card 4: Preferences & Settings (Shared)
        card_opts, opts_vbox = make_card("System Preferences & Safety", "#00f0ff")
        opts_layout = QGridLayout()
        opts_layout.setSpacing(6)

        self.chk_typing = QCheckBox("Ignore knocks while typing")
        self.chk_typing.setChecked(self.config.get("suppress_typing", True))
        self.chk_typing.toggled.connect(lambda val: self.config.set("suppress_typing", val))

        self.chk_sides = QCheckBox("Distinguish Left vs Right chassis")
        self.chk_sides.setChecked(self.config.get("detect_sides", True))
        self.chk_sides.toggled.connect(lambda val: self.config.set("detect_sides", val))

        self.chk_crowded = QCheckBox("Crowded environment (require left/right chassis taps)")
        self.chk_crowded.setChecked(self.config.get("crowded_mode", False))
        self.chk_crowded.toggled.connect(self.on_crowded_mode_changed)

        self.chk_feedback = QCheckBox("Audio click sound on trigger")
        self.chk_feedback.setChecked(self.config.get("enable_sound_feedback", True))
        self.chk_feedback.toggled.connect(lambda val: self.config.set("enable_sound_feedback", val))

        self.chk_notif = QCheckBox("Desktop notifications")
        self.chk_notif.setChecked(self.config.get("enable_notifications", True))
        self.chk_notif.toggled.connect(lambda val: self.config.set("enable_notifications", val))

        self.chk_wake_word = QCheckBox("Ask V1M to require a 'Laptop' wake phrase")
        self.chk_wake_word.setChecked(self.config.get("voice_require_wake_word", True))
        self.chk_wake_word.toggled.connect(self.on_wake_word_changed)

        self.chk_autostart = QCheckBox("Start automatically with Windows")
        self.chk_autostart.setChecked(self.config.get("start_with_windows", False))
        self.chk_autostart.toggled.connect(self.on_autostart_changed)

        opts_layout.addWidget(self.chk_typing, 0, 0)
        opts_layout.addWidget(self.chk_sides, 1, 0)
        opts_layout.addWidget(self.chk_crowded, 2, 0)
        opts_layout.addWidget(self.chk_feedback, 3, 0)
        opts_layout.addWidget(self.chk_notif, 4, 0)
        opts_layout.addWidget(self.chk_wake_word, 5, 0)
        opts_layout.addWidget(self.chk_autostart, 6, 0)
        opts_vbox.addLayout(opts_layout)
        scroll_content_layout.addWidget(card_opts)

        # Card 5: v1m Cloud Intent Verification (guardrail & intent resolver)
        card_v1m, v1m_vbox = make_card(
            "☁ v1m Cloud Intent Verification (System-One Guardrail)", "#fbbf24"
        )

        self.chk_v1m = QCheckBox("Enable v1m Cloud Intent Verification")
        self.chk_v1m.setChecked(self.config.get("enable_v1m_verification", False))
        self.chk_v1m.toggled.connect(self.on_v1m_toggled)
        self._update_v1m_tooltip()
        v1m_vbox.addWidget(self.chk_v1m)

        v1m_hint = QLabel(
            "Windows first matches speech against the supported command phrases and "
            "sends the recognized phrase and alternatives to V1M for intent and action "
            "classification. Requests with different confidence/context are evaluated "
            "separately. When V1M is enabled, no command runs without its decision."
        )
        v1m_hint.setWordWrap(True)
        v1m_hint.setStyleSheet("color: #64748b; font-size: 10px; background: transparent;")
        v1m_vbox.addWidget(v1m_hint)

        prob_row = QHBoxLayout()
        prob_lbl = QLabel("Minimum V1M Valid-Command Probability:")
        prob_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        curr_prob = int(round(float(self.config.get("v1m_min_probability", 0.55)) * 100))
        self.v1m_prob_val = QLabel(f"{curr_prob}%")
        self.v1m_prob_val.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 11px; background: transparent;")
        prob_row.addWidget(prob_lbl)
        prob_row.addStretch()
        prob_row.addWidget(self.v1m_prob_val)
        v1m_vbox.addLayout(prob_row)

        self.v1m_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self.v1m_prob_slider.setRange(20, 95)
        self.v1m_prob_slider.setValue(curr_prob)
        self.v1m_prob_slider.valueChanged.connect(self.on_v1m_prob_changed)
        v1m_vbox.addWidget(self.v1m_prob_slider)

        risk_row = QHBoxLayout()
        risk_lbl = QLabel("Maximum V1M Execution Risk (0–4):")
        risk_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        curr_risk = int(round(float(self.config.get("v1m_max_execution_risk", 4.0)) * 10))
        self.v1m_risk_val = QLabel(f"{curr_risk / 10:.1f}" + (" (off)" if curr_risk >= 40 else ""))
        self.v1m_risk_val.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 11px; background: transparent;")
        risk_row.addWidget(risk_lbl)
        risk_row.addStretch()
        risk_row.addWidget(self.v1m_risk_val)
        v1m_vbox.addLayout(risk_row)

        self.v1m_risk_slider = QSlider(Qt.Orientation.Horizontal)
        self.v1m_risk_slider.setRange(0, 40)
        self.v1m_risk_slider.setValue(curr_risk)
        self.v1m_risk_slider.setToolTip(
            "V1M estimates accidental execution risk from 0 (lowest) to 4 (highest). "
            "A command is blocked when its score exceeds this limit. 4 disables this gate."
        )
        self.v1m_risk_slider.valueChanged.connect(self.on_v1m_risk_changed)
        v1m_vbox.addWidget(self.v1m_risk_slider)

        key_row = QHBoxLayout()
        key_lbl = QLabel("API Key:")
        key_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px; background: transparent;")
        self.v1m_key_input = QLineEdit(str(self.config.get("v1m_api_key") or ""))
        self.v1m_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.v1m_key_input.setPlaceholderText(
            "V1M_API_KEY is set in the environment (field optional)"
            if os.environ.get("V1M_API_KEY", "").strip()
            else "Paste a v1m API key (or set V1M_API_KEY)"
        )
        self.v1m_key_input.setAccessibleName("v1m API key")
        self.v1m_key_input.setToolTip(
            "Stored in the existing ~/.laptoptap/config.json settings file. "
            "The V1M_API_KEY environment "
            "variable always takes precedence when present."
        )
        self.v1m_key_input.editingFinished.connect(self._save_v1m_key)
        self.btn_v1m_test = QPushButton("Test Connection")
        self.btn_v1m_test.clicked.connect(self.run_v1m_connection_test)
        key_row.addWidget(key_lbl)
        key_row.addWidget(self.v1m_key_input, 1)
        key_row.addWidget(self.btn_v1m_test)
        v1m_vbox.addLayout(key_row)

        self.v1m_status = QLabel(self._v1m_idle_status())
        self.v1m_status.setWordWrap(True)
        self.v1m_status.setStyleSheet(
            "color: #94a3b8; font-size: 10px; background: transparent;"
        )
        v1m_vbox.addWidget(self.v1m_status)

        self.btn_v1m_exchange = QPushButton("View Last V1M Request / Response")
        self.btn_v1m_exchange.clicked.connect(self.show_last_v1m_exchange)
        self.btn_v1m_exchange.setToolTip(
            "Inspect the last cloud request body and V1M response. API credentials are omitted."
        )
        v1m_vbox.addWidget(self.btn_v1m_exchange)
        scroll_content_layout.addWidget(card_v1m)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 1)

        # 5. Footer Controls Bar (ALWAYS visible at the very bottom!)
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 3, 0, 0)
        footer_layout.setSpacing(8)

        self.btn_calibrate = QPushButton("🎯 Calibrate Taps")
        self.btn_calibrate.setStyleSheet("""
            QPushButton { background-color: #1e1b4b; border: 1px solid #6366f1; color: #c7d2fe; font-weight: bold; }
            QPushButton:hover { background-color: #312e81; border: 1px solid #818cf8; }
        """)
        self.btn_calibrate.clicked.connect(self.run_active_calibration)

        self.btn_toggle = QPushButton("Pause")
        self.btn_toggle.clicked.connect(self.toggle_monitoring)

        self.btn_minimize = QPushButton("Hide to Tray")
        self.btn_minimize.setStyleSheet("""
            QPushButton { background-color: #00f0ff; color: #0b0d13; font-weight: bold; border: none; }
            QPushButton:hover { background-color: #38bdf8; }
        """)
        self.btn_minimize.clicked.connect(self.hide)

        footer_layout.addWidget(self.btn_calibrate, 1)
        footer_layout.addWidget(self.btn_toggle, 1)
        footer_layout.addWidget(self.btn_minimize, 1)
        main_layout.addLayout(footer_layout)

    def switch_mode(self, mode: str):
        if mode not in ("knock", "voice"):
            mode = "knock"
        if self.config.get("active_mode") != mode:
            self.config.set("active_mode", mode)

        if mode == "knock":
            self.btn_mode_knock.setChecked(True)
            self.btn_mode_voice.setChecked(False)
            self.btn_mode_knock.setStyleSheet("""
                QPushButton {
                    background-color: #00f0ff;
                    color: #0b0d13;
                    border: none;
                    font-weight: 700;
                    border-radius: 6px;
                }
            """)
            self.btn_mode_voice.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    border: none;
                    font-weight: 600;
                }
                QPushButton:hover { color: #ffffff; }
            """)

            self.stacked_widget.setCurrentWidget(self.panel_knock)
            self._set_status_badge("● Starting Knock Mode…", "#00f0ff", "#0c2b32")
            self.btn_calibrate.setText("🎯 Calibrate Taps")
            self.btn_calibrate.setStyleSheet("""
                QPushButton { background-color: #1e1b4b; border: 1px solid #6366f1; color: #c7d2fe; font-weight: bold; }
                QPushButton:hover { background-color: #312e81; border: 1px solid #818cf8; }
            """)
            self.event_badge.setText("Knock 1, 2, or 3 times on the laptop body to trigger shortcuts...")

            if self.voice_detector and self.voice_detector.is_running:
                self.voice_detector.stop()
            self.tap_detector.start()
            if not self.is_monitoring_paused and not self.audio_engine.is_running:
                if self.audio_engine.start():
                    self._sync_active_microphone()
                    self._set_status_badge("● Knock Mode Active", "#00f0ff", "#0c2b32")
                else:
                    self._show_engine_error(self.audio_engine.last_start_error or "Microphone could not be opened.")
            elif self.audio_engine.is_running:
                self._set_status_badge("● Knock Mode Active", "#00f0ff", "#0c2b32")

        elif mode == "voice":
            self.btn_mode_voice.setChecked(True)
            self.btn_mode_knock.setChecked(False)
            self.btn_mode_voice.setStyleSheet("""
                QPushButton {
                    background-color: #a855f7;
                    color: #ffffff;
                    border: none;
                    font-weight: 700;
                    border-radius: 6px;
                }
            """)
            self.btn_mode_knock.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    border: none;
                    font-weight: 600;
                }
                QPushButton:hover { color: #ffffff; }
            """)

            self.stacked_widget.setCurrentWidget(self.panel_voice)
            self._set_status_badge("● Starting Voice Mode…", "#c084fc", "#261238")
            self.btn_calibrate.setText("🎙️ Calibrate Voice")
            self.btn_calibrate.setStyleSheet("""
                QPushButton { background-color: #2e1065; border: 1px solid #a855f7; color: #e9d5ff; font-weight: bold; }
                QPushButton:hover { background-color: #3b0764; border: 1px solid #c084fc; }
            """)
            wake = "Laptop Copy" if self.config.get("voice_require_wake_word", True) else "Copy"
            self.event_badge.setText(f"Starting speech-to-text… Try saying '{wake}' when ready.")

            if self.audio_engine.is_running:
                self.audio_engine.stop()
            self.tap_detector.pause()
            if not self.is_monitoring_paused and self.voice_detector and not self.voice_detector.is_running:
                if not self.voice_detector.start():
                    self._show_engine_error(self.voice_detector.last_error or "Voice engine could not be started.")
        QTimer.singleShot(0, lambda: self.scroll_area.verticalScrollBar().setValue(0))

    def _populate_audio_devices(self):
        self.combo_mic.clear()
        self.audio_devices = AudioEngine.get_input_devices()
        cur_dev = self.config.get("selected_input_device")
        cur_sig = self.config.get("selected_input_device_signature")
        resolved = AudioEngine.resolve_device_index(cur_dev, cur_sig)
        for dev in self.audio_devices:
            self.combo_mic.addItem(f"{dev['name']} — {dev['hostapi']} (#{dev['index']})", dev['index'])
        idx = self.combo_mic.findData(resolved)
        if idx >= 0:
            self.combo_mic.setCurrentIndex(idx)

    def on_device_changed(self, idx):
        if idx < 0:
            return
        dev_idx = self.combo_mic.itemData(idx)
        dev = next((item for item in self.audio_devices if item["index"] == dev_idx), None)
        signature = dev.get("signature") if dev else None
        if self.audio_engine.change_device(dev_idx, signature):
            self.config.update({
                "selected_input_device": dev_idx,
                "selected_input_device_signature": signature,
            })
            if self.audio_engine.is_running:
                self._set_status_badge("● Knock Mode Active", "#00f0ff", "#0c2b32")
        else:
            self._show_engine_error(self.audio_engine.last_start_error or "The selected microphone could not be opened.")
            self._sync_active_microphone()

    def _sync_active_microphone(self):
        dev_idx = self.audio_engine.active_device_index
        if dev_idx is None:
            return
        idx = self.combo_mic.findData(dev_idx)
        if idx >= 0:
            self.combo_mic.blockSignals(True)
            self.combo_mic.setCurrentIndex(idx)
            self.combo_mic.blockSignals(False)
        self.config.update({
            "selected_input_device": dev_idx,
            "selected_input_device_signature": self.audio_engine.active_device_signature,
        })

    def _set_status_badge(self, text, color, background):
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"font-size: 10px; font-weight: bold; color: {color}; "
            f"background-color: {background}; padding: 4px 10px; "
            f"border-radius: 10px; border: 1px solid {color};"
        )

    def _show_engine_error(self, message):
        message = str(message or "Unknown engine error")
        self._set_status_badge("● Engine Error", "#fb7185", "#3b111b")
        self.event_badge.setText(f"⚠ {message}")
        self.event_badge.setStyleSheet(
            "background-color: #3b111b; border: 1px solid #fb7185; border-radius: 6px; "
            "padding: 6px; font-size: 10px; font-weight: 600; color: #fecdd3;"
        )

    def _populate_actions(self, combo, current_key):
        combo.clear()
        for key, label in ACTION_LABELS.items():
            combo.addItem(label, key)
        idx = combo.findData(current_key)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _save_action(self, tap_key, combo):
        act = combo.currentData()
        if not act:
            return
        # Build a NEW dict: never mutate the shared dict in place while the
        # audio worker thread may be reading it. set() persists atomically and
        # notifies listeners, so the active dispatcher picks this up on the
        # very next knock — no restart needed.
        actions = dict(self.config.get("actions", {}) or {})
        if actions.get(tap_key) == act:
            return
        actions[tap_key] = act
        self.config.set("actions", actions)

    def _refresh_from_config(self):
        """Sync widgets with the live config.

        Runs on the GUI thread (via SignalBridge, even when the change was
        detected on the audio thread) after any config change — our own or an
        external process such as calibrate.py. blockSignals prevents these
        programmatic updates from re-triggering save handlers.
        """
        actions = self.config.get("actions", {}) or {}
        for combo, key in getattr(self, "_action_combo_map", []):
            idx = combo.findData(actions.get(key))
            if idx >= 0 and idx != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        voice_actions = self.config.get("voice_actions", {}) or {}
        for combo, keys in getattr(self, "_voice_combo_map", []):
            idx = combo.findData(voice_actions.get(keys[0]))
            if idx >= 0 and idx != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        sens = int(round(float(self.config.get("sensitivity", 0.90)) * 100))
        if sens != self.sens_slider.value():
            self.sens_slider.blockSignals(True)
            self.sens_slider.setValue(sens)
            self.sens_slider.blockSignals(False)
            self.sens_val_label.setText(f"{sens}%")

        vconf = int(round(float(self.config.get("voice_confidence_threshold", 0.55)) * 100))
        if vconf != self.voice_sens_slider.value():
            self.voice_sens_slider.blockSignals(True)
            self.voice_sens_slider.setValue(vconf)
            self.voice_sens_slider.blockSignals(False)
            self.voice_sens_val_label.setText(f"{vconf}%")

        for checkbox, key in (
            (self.chk_typing, "suppress_typing"),
            (self.chk_sides, "detect_sides"),
            (self.chk_crowded, "crowded_mode"),
            (self.chk_feedback, "enable_sound_feedback"),
            (self.chk_notif, "enable_notifications"),
            (self.chk_wake_word, "voice_require_wake_word"),
            (self.chk_autostart, "start_with_windows"),
            (self.chk_v1m, "enable_v1m_verification"),
        ):
            want = bool(self.config.get(key))
            if checkbox.isChecked() != want:
                checkbox.blockSignals(True)
                checkbox.setChecked(want)
                checkbox.blockSignals(False)
        self._update_v1m_tooltip()
        prob_val = int(round(float(self.config.get("v1m_min_probability", 0.55)) * 100))
        if hasattr(self, "v1m_prob_slider") and prob_val != self.v1m_prob_slider.value():
            self.v1m_prob_slider.blockSignals(True)
            self.v1m_prob_slider.setValue(prob_val)
            self.v1m_prob_slider.blockSignals(False)
            if hasattr(self, "v1m_prob_val"):
                self.v1m_prob_val.setText(f"{prob_val}%")

    def on_sensitivity_changed(self, val):
        sens = val / 100.0
        self.sens_val_label.setText(f"{val}%")
        # flush=False: the detector reads sensitivity live from memory, and the
        # debounced save turns a slider drag into ONE atomic disk write instead
        # of ~60 fsyncs/second.
        self.config.set("sensitivity", sens, flush=False)

    def on_voice_sensitivity_changed(self, val):
        thresh = round(val / 100.0, 2)
        self.voice_sens_val_label.setText(f"{val}%")
        # flush=False: VoiceDetector's config listener debounces the respawn,
        # so a slider drag produces ONE restart after the drag settles.
        self.config.set("voice_confidence_threshold", thresh, flush=False)

    def _save_voice_action(self, cmd_keys, combo):
        """Persist the action associated with a V1M command choice."""
        act = combo.currentData()
        if not act:
            return
        # New dict (never mutate in place), same pattern as _save_action:
        # set() persists atomically and notifies listeners.
        voice_actions = dict(self.config.get("voice_actions", {}) or {})
        if all(voice_actions.get(k) == act for k in cmd_keys):
            return
        for k in cmd_keys:
            voice_actions[k] = act
        self.config.set("voice_actions", voice_actions)

    def _refresh_voice_instructions(self):
        wake_required = bool(self.config.get("voice_require_wake_word", True))
        if wake_required:
            self.voice_desc.setText(
                "🎙️ Say one of the listed commands, preferably starting with 'Laptop' "
                "(for example, 'Laptop, copy that'). Windows uses the command vocabulary "
                "to improve recognition; V1M decides whether and which action to run."
            )
        else:
            self.voice_desc.setText(
                "🎙️ Say one of the listed commands. Windows uses the command vocabulary "
                "to improve recognition; V1M decides whether and which action to run."
            )
        prefix = "Laptop " if wake_required else ""
        for label, phrase in self.voice_phrase_labels:
            variants = " / ".join(prefix + item.strip() for item in phrase.split(" / "))
            label.setText(f'🗣️ "{variants}"')

    def on_wake_word_changed(self, enabled):
        self.config.set("voice_require_wake_word", bool(enabled))
        self._refresh_voice_instructions()
        # The setting is included in the V1M request; respawning the speech
        # helper is unnecessary, but the listener may restart it harmlessly.

    def on_crowded_mode_changed(self, enabled):
        values = {"crowded_mode": bool(enabled)}
        if enabled:
            values["detect_sides"] = True
            self.chk_sides.blockSignals(True)
            self.chk_sides.setChecked(True)
            self.chk_sides.blockSignals(False)
            self.event_badge.setText(
                "Crowded mode enabled: tap the left or right side of the chassis."
            )
        self.config.update(values)

    def on_autostart_changed(self, enabled):
        if not self.config.set("start_with_windows", bool(enabled)):
            self.chk_autostart.blockSignals(True)
            self.chk_autostart.setChecked(not enabled)
            self.chk_autostart.blockSignals(False)
            self._show_engine_error(f"Could not update Windows startup: {self.config.last_error or 'access denied'}")

    # ------------------------------------------------------------------
    # v1m cloud intent verification
    # ------------------------------------------------------------------

    def _update_v1m_tooltip(self):
        if not hasattr(self, "chk_v1m"):
            return
        try:
            min_p = float(self.config.get("v1m_min_probability", 0.55))
        except (TypeError, ValueError):
            min_p = 0.55
        pct = int(round(min_p * 100))
        self.chk_v1m.setToolTip(
            f"Speech recognition confidence is separate from V1M's intent probability. "
            f"V1M sends is_valid_command (minimum {pct}%), action and execution_risk "
            f"for each recognized phrase before it may "
            f"trigger an action. No command runs if V1M is "
            f"unavailable or times out while enabled."
        )

    def on_v1m_prob_changed(self, val):
        if hasattr(self, "v1m_prob_val"):
            self.v1m_prob_val.setText(f"{val}%")
        prob = round(val / 100.0, 2)
        self.config.set("v1m_min_probability", prob, flush=False)
        self._update_v1m_tooltip()
        if hasattr(self, "v1m_status") and not self.btn_v1m_test.text().startswith("Testing"):
            self.v1m_status.setText(self._v1m_idle_status())

    def on_v1m_risk_changed(self, val):
        if hasattr(self, "v1m_risk_val"):
            risk = val / 10.0
            self.v1m_risk_val.setText(f"{risk:.1f}" + (" (off)" if val >= 40 else ""))
        self.config.set("v1m_max_execution_risk", round(val / 10.0, 1), flush=False)

    def _v1m_idle_status(self):
        """Static status line for the cloud-verification card (no secrets)."""
        if not self.voice_verifier:
            return "V1M unavailable — commands are blocked while verification is enabled."
        info = self.voice_verifier.describe()
        if not info["sdk_installed"]:
            return f"⚠ typesafe-sdk not installed ({info['endpoint']}) — commands are blocked while V1M is enabled."
        min_pct = int(round(info.get("min_probability", 0.55) * 100))
        if info["key_source"] == "missing":
            return (
                f"Ready · {info['endpoint']} · model {info['model']} · "
                f"min intent {min_pct}% · no API key yet."
            )
        return (
            f"Ready · {info['endpoint']} · model {info['model']} · "
            f"min intent {min_pct}% · key from {info['key_source']} · wait cap "
            f"{info['wait_timeout_ms']:.0f} ms."
        )

    def on_v1m_toggled(self, enabled):
        self.config.set("enable_v1m_verification", bool(enabled))
        if not enabled:
            self.v1m_status.setText("Disabled — recognised phrases are not sent to the cloud.")
            self.v1m_status.setStyleSheet(
                "color: #94a3b8; font-size: 10px; background: transparent;"
            )
            return
        if self.voice_verifier:
            self.voice_verifier.start()
        if not (self.voice_verifier and self.voice_verifier.api_key()):
            self.v1m_status.setText("⚠ Enabled, but no API key — commands will be blocked until V1M can verify them.")
            self.v1m_status.setStyleSheet(
                "color: #fbbf24; font-size: 10px; font-weight: 600; background: transparent;"
            )
        else:
            self.v1m_status.setText("Enabled. Run a connection test to confirm the key.")
            self.v1m_status.setStyleSheet(
                "color: #94a3b8; font-size: 10px; background: transparent;"
            )

    def _save_v1m_key(self):
        key = self.v1m_key_input.text().strip()
        if str(self.config.get("v1m_api_key") or "") == key:
            return
        # Live config: the verifier re-reads the key on the very next phrase.
        self.config.set("v1m_api_key", key)

    def run_v1m_connection_test(self):
        """Probe the endpoint off the GUI thread; the result arrives via signal."""
        if not self.voice_verifier:
            self.v1m_status.setText("Cloud verifier unavailable in this build.")
            return
        self._save_v1m_key()
        self.btn_v1m_test.setEnabled(False)
        self.v1m_status.setText(f"Testing {self.voice_verifier.endpoint} …")
        self.v1m_status.setStyleSheet(
            "color: #94a3b8; font-size: 10px; background: transparent;"
        )

        def _worker():
            ok, message = self.voice_verifier.test_connection()
            self.bridge.v1m_test_signal.emit(ok, message)

        threading.Thread(target=_worker, name="v1m-connection-test", daemon=True).start()

    def on_v1m_test_result(self, ok, message):
        self.btn_v1m_test.setEnabled(True)
        color = "#4ade80" if ok else "#fb7185"
        prefix = "✓ " if ok else "✗ "
        self.v1m_status.setText(prefix + message)
        self.v1m_status.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: 600; background: transparent;"
        )
        if not ok:
            # Surfaced in the badge too: a failed test otherwise looks like a
            # silent no-op if the settings card is scrolled out of view.
            self.event_badge.setText(f"⚠ v1m connection test failed: {message}")
            self.event_badge.setStyleSheet(
                "background-color: #3b111b; border: 1px solid #fb7185; border-radius: 6px; "
                "padding: 6px; font-size: 10px; font-weight: 600; color: #fecdd3;"
            )

    def show_last_v1m_exchange(self):
        """Show the most recent request body and raw/parsed V1M response."""
        exchange = self.voice_verifier.last_exchange_json() if self.voice_verifier else None
        dialog = QDialog(self)
        dialog.setWindowTitle("Last V1M Request / Response")
        dialog.resize(820, 640)

        layout = QVBoxLayout(dialog)
        details = QPlainTextEdit(dialog)
        details.setReadOnly(True)
        details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        details.setPlainText(
            exchange or "No V1M cloud request has completed in this session yet."
        )
        details.setStyleSheet(
            "QPlainTextEdit { background: #0b0d13; color: #dbeafe; "
            "border: 1px solid #293044; font-family: Consolas, monospace; font-size: 10px; }"
        )
        layout.addWidget(details, 1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    def on_voice_status_changed(self, state, message):
        if self.config.get("active_mode", "knock") != "voice":
            return
        if state == "ready":
            self._set_status_badge("● Voice Mode Active", "#c084fc", "#261238")
            wake = "Laptop Copy" if self.config.get("voice_require_wake_word", True) else "Copy"
            self.event_badge.setText(f"Listening for speech. Try saying '{wake}'.")
        elif state == "starting":
            self._set_status_badge("● Starting Voice Mode…", "#c084fc", "#261238")
        elif state == "error":
            self._show_engine_error(message)

    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(self)

        from pathlib import Path
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
        else:
            from PyQt6.QtGui import QPixmap
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.transparent)
            p = QPainter(pixmap)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(QBrush(QColor(0, 240, 255)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(4, 4, 24, 24)
            p.end()
            icon = QIcon(pixmap)

        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("عقده ای — میانبرهای ضربه و صدا")

        tray_menu = QMenu()
        act_show = QAction("Open Dashboard", self)
        act_show.triggered.connect(self.show_normal_and_raise)
        tray_menu.addAction(act_show)

        act_pause = QAction("Pause / Resume", self)
        act_pause.triggered.connect(self.toggle_monitoring)
        tray_menu.addAction(act_pause)

        tray_menu.addSeparator()
        act_quit = QAction("Exit", self)
        act_quit.triggered.connect(self.clean_quit)
        tray_menu.addAction(act_quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def show_normal_and_raise(self):
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
        self.raise_()
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_normal_and_raise()

    def toggle_monitoring(self):
        self.is_monitoring_paused = not self.is_monitoring_paused
        mode = self.config.get("active_mode", "knock")

        if self.is_monitoring_paused:
            if self.audio_engine.is_running:
                self.audio_engine.stop()
            if self.voice_detector and self.voice_detector.is_running:
                self.voice_detector.stop()
            self.status_badge.setText("⏸ Monitoring Paused")
            self.status_badge.setStyleSheet("font-size: 10px; font-weight: bold; color: #ffab00; background-color: #2b2210; padding: 4px 10px; border-radius: 10px; border: 1px solid #ffab00;")
            self.btn_toggle.setText("Resume")
        else:
            self.btn_toggle.setText("Pause")
            self.switch_mode(mode)

    def on_level_received(self, peak, threshold):
        self.current_peak = peak
        self.current_threshold = threshold

    def refresh_display(self):
        if not self.isVisible():
            return
        if self.config.get("active_mode", "knock") == "knock":
            history = self.audio_engine.get_waveform_history()
            self.waveform_widget.update_data(history, self.current_peak, self.current_threshold)
            if self.audio_engine.last_callback_error:
                error = self.audio_engine.last_callback_error
                self.audio_engine.last_callback_error = None
                self._show_engine_error(f"Audio processing failed: {error}")
            elif self.audio_engine.dropped_blocks:
                self._set_status_badge(
                    f"● Knock Active — {self.audio_engine.dropped_blocks} dropped blocks",
                    "#fbbf24",
                    "#35260d",
                )

    def on_gesture_received(self, gesture: DetectedTapGesture):
        if self.config.get("active_mode", "knock") != "knock":
            return

        self.waveform_widget.flash()

        # Live lookup through the shared resolver: mapping changes made in the
        # dropdowns take effect on this very gesture.
        action_name = resolve_gesture_action(self.config, gesture)
        action_title = ACTION_LABELS.get(action_name, action_name)
        side_text = f" ({gesture.side.capitalize()} side)" if gesture.side != "center" else ""
        text = f"⚡ {gesture.count}× Knock{side_text} ➔ {action_title}"

        self.event_badge.setText(text)
        self.event_badge.setStyleSheet("""
            background-color: #0b2f3a;
            border: 1px solid #00f0ff;
            border-radius: 6px;
            padding: 6px;
            font-size: 11px;
            font-weight: bold;
            color: #00f0ff;
        """)

        if self.config.get("enable_notifications", True) and not self.isVisible():
            self.tray_icon.showMessage("عقده ای · ضربه", text, QSystemTrayIcon.MessageIcon.Information, 1200)

        if action_name and action_name != "none" and not self.executor.trigger(action_name):
            self._show_engine_error(f"Action failed: {action_title}")

    def _on_voice_captured(self, cmd, confidence, spoken_text="", alternates=None):
        """v1m guardrail hook for the recogniser's stdout reader thread.

        Runs BEFORE the Qt hand-off so the network (when it is used at all)
        never touches the GUI thread, and returns within
        ``v1m_wait_timeout_ms`` even when the cloud is down or slow.
        """
        if not self.voice_verifier:
            self.bridge.voice_signal.emit(cmd, confidence, spoken_text or cmd, None)
            return
        decision = self.voice_verifier.verify(
            cmd, spoken_text, confidence, alternates=alternates
        )
        if not decision.allow:
            self.bridge.voice_reject_signal.emit(decision)
            return
        # decision.command may differ from cmd when the model resolved the intent.
        self.bridge.voice_signal.emit(decision.command, confidence, spoken_text or cmd, decision)

    def on_voice_rejected(self, decision):
        """Badge/notification for a phrase the v1m guardrail vetoed."""
        if self.config.get("active_mode", "knock") != "voice":
            return
        phrase = decision.phrase or decision.command
        text = f"🚫 v1m blocked \"{phrase}\" — {decision.reason}"
        self.event_badge.setText(text)
        self.event_badge.setStyleSheet("""
            background-color: #3b111b;
            border: 1px solid #fb7185;
            border-radius: 6px;
            padding: 6px;
            font-size: 11px;
            font-weight: bold;
            color: #fecdd3;
        """)
        if self.config.get("enable_notifications", True) and not self.isVisible():
            self.tray_icon.showMessage(
                "عقده ای · صدا", text,
                QSystemTrayIcon.MessageIcon.Warning, 1500,
            )

    def on_voice_received(self, word: str, confidence: float, spoken_text: str = "",
                          decision=None):
        if self.config.get("active_mode", "knock") != "voice":
            return

        action_name = resolve_voice_action(self.config, word)
        if not action_name:
            self._show_engine_error(f"Unmapped voice command was ignored: {word}")
            return

        action_title = ACTION_LABELS.get(action_name, action_name)
        display_text = spoken_text if spoken_text else word
        guard = ""
        if decision is not None and getattr(decision, "is_cloud", False):
            try:
                minimum = float(self.config.get("v1m_min_probability", 0.55))
            except (TypeError, ValueError):
                minimum = 0.55
            guard = (
                f" · V1M intent {decision.probability:.0%} "
                f"(min {minimum:.0%})"
            )
            action_confidence = getattr(decision, "action_confidence", None)
            if action_confidence is not None:
                guard += f" · action {action_confidence:.0%}"
            execution_risk = getattr(decision, "execution_risk", None)
            if execution_risk is not None:
                guard += f" · risk {execution_risk:.1f}/4"
        text = (f"🎙️ Spoken: \"{display_text.upper()}\" ({int(confidence*100)}%){guard} "
                f"➔ {action_title}")
        self.event_badge.setToolTip(
            "Speech % = local speech-recognition confidence. "
            "V1M intent % = cloud probability that this is a valid command; "
            "the minimum is shown beside it. Action % = V1M Choice confidence "
            "for the selected action; it does not replace the intent gate. "
            "Risk is V1M's 0–4 execution-risk score."
        )

        self.event_badge.setText(text)
        self.event_badge.setStyleSheet("""
            background-color: #2b123d;
            border: 1px solid #a855f7;
            border-radius: 6px;
            padding: 6px;
            font-size: 11px;
            font-weight: bold;
            color: #f3e8ff;
        """)

        if self.config.get("enable_notifications", True) and not self.isVisible():
            self.tray_icon.showMessage("عقده ای · صدا", text, QSystemTrayIcon.MessageIcon.Information, 1200)

        if action_name and action_name != "none" and not self.executor.trigger(action_name):
            self._show_engine_error(f"Action failed: {action_title}")

    def run_active_calibration(self):
        mode = self.config.get("active_mode", "knock")
        if mode == "knock":
            self.run_tap_calibration()
        else:
            self.run_voice_calibration()

    def run_tap_calibration(self):
        import subprocess
        from pathlib import Path
        if self.audio_engine and self.audio_engine.is_running:
            self.audio_engine.stop()
        script = str(Path(__file__).parent / "calibrate.py")
        if sys.platform == "win32" and not getattr(sys, "frozen", False):
            subprocess.Popen(
                [sys.executable, script],
                cwd=str(Path(__file__).parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            self._show_engine_error("Calibration launcher is available from source mode on Windows.")

    def run_voice_calibration(self):
        import subprocess
        from pathlib import Path
        if self.voice_detector and self.voice_detector.is_running:
            self.voice_detector.stop()
        script = str(Path(__file__).parent / "voice_calibrate.py")
        if sys.platform == "win32" and not getattr(sys, "frozen", False):
            subprocess.Popen(
                [sys.executable, script],
                cwd=str(Path(__file__).parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            self._show_engine_error("Voice calibration launcher is available from source mode on Windows.")

    def changeEvent(self, event):
        super().changeEvent(event)
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            new_conf = self.config.get("voice_confidence_threshold", 0.55)
            if hasattr(self, 'voice_sens_slider'):
                self.voice_sens_slider.blockSignals(True)
                self.voice_sens_slider.setValue(int(new_conf * 100))
                self.voice_sens_val_label.setText(f"{int(new_conf * 100)}%")
                self.voice_sens_slider.blockSignals(False)
            mode = self.config.get("active_mode", "knock")
            if not self.is_monitoring_paused:
                if mode == "voice" and self.voice_detector and not self.voice_detector.is_running:
                    if not self.voice_detector.start():
                        self._show_engine_error(self.voice_detector.last_error)
                elif mode == "knock" and self.audio_engine and not self.audio_engine.is_running:
                    self.tap_detector.start()
                    if self.audio_engine.start():
                        self._sync_active_microphone()
                    else:
                        self._show_engine_error(self.audio_engine.last_start_error)

    def closeEvent(self, event):
        if self.tray_icon.isVisible():
            self.hide()
            if self.config.get("enable_notifications", True):
                self.tray_icon.showMessage(
                    "عقده ای در پس‌زمینه فعال است",
                    "برنامه در system tray در حال اجراست.",
                    QSystemTrayIcon.MessageIcon.Information,
                    1500
                )
            event.ignore()
        else:
            self.clean_quit()

    def clean_quit(self):
        self.audio_engine.stop()
        if self.voice_detector:
            self.voice_detector.stop()
        if self.voice_verifier:
            self.voice_verifier.shutdown()
        self.tap_detector.stop()
        self.config.flush()  # persist any debounced slider edits
        self.tray_icon.hide()
        QApplication.quit()

    def hideEvent(self, event):
        # Stop the 28 ms display timer while hidden (tray): a minimized
        # instance should not copy/repaint a waveform nobody sees.
        super().hideEvent(event)
        timer = getattr(self, "timer", None)
        if timer is not None:
            timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        timer = getattr(self, "timer", None)
        if timer is not None and not timer.isActive():
            timer.start(28)
