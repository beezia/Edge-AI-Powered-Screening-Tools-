from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QMovie, QPixmap
import pyqtgraph as pg
import sys
import time
import random
import math


class EdgeAIDemoUI:
    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.app = QApplication(sys.argv)

        self.window = QWidget()
        self.window.setWindowTitle("Edge AI: Screening Tools powered by CPU+iGPU")
        self.window.resize(1100, 700)

        # -------------------------------
        # GLOBAL DARK THEME
        # -------------------------------
        self.window.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #E0E0E0;
                font-family: Segoe UI, Arial;
            }
            QLabel {
                color: #E0E0E0;
            }
            QProgressBar {
                border: 1px solid #2A2A2A;
                border-radius: 5px;
                background-color: #1E1E1E;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 5px;
            }
        """)

        outer_layout = QVBoxLayout()
        self.window.setLayout(outer_layout)

        # -------------------------------
        # HEADER (Title + Logo)
        # -------------------------------
        header_layout = QHBoxLayout()
        outer_layout.addLayout(header_layout)

        title_layout = QVBoxLayout()

        self.title_label = QLabel("Edge AI: Screening Tools powered by CPU+iGPU")
       # self.title_label.setAlignment(Qt.AlignCenter)  # <-- this centers it horizontally
        self.title_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel(
            "Scale throughput, reduce latency, protect margin — without scaling infrastructure"
        )
        #self.subtitle_label.setAlignment(Qt.AlignCenter)  # <-- this centers it horizontally
        self.subtitle_label.setStyleSheet(
            "font-size: 20px; font-style: italic; color: #A0A0A0;"
        )
        title_layout.addWidget(self.subtitle_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        # Company Logo (Top Right)
        self.logo_label = QLabel()
        logo_pixmap = QPixmap("assets/logo.png")
        self.logo_label.setPixmap(logo_pixmap.scaled(
            280, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header_layout.addWidget(self.logo_label)

        # -------------------------------
        # MAIN CONTENT
        # -------------------------------
        main_layout = QHBoxLayout()
        outer_layout.addLayout(main_layout)

        # LEFT SIDE
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 3)

        # GIF
        self.sample_gif_label = QLabel()
        self.sample_gif_label.setAlignment(Qt.AlignCenter)
        self.sample_gif_label.setMinimumHeight(400)
        left_layout.addWidget(self.sample_gif_label, 1)

        self.sample_movie = QMovie("assets/sample_flow.gif")
        self.sample_movie.setScaledSize(QSize(580, 400))
        self.sample_gif_label.setMovie(self.sample_movie)
        self.sample_movie.start()

        # Dark Plot Styling
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#E0E0E0')

        self.curve_plot = pg.PlotWidget(title="Amplification Curve")
        self.curve_plot.showGrid(x=True, y=True, alpha=0.2)
        left_layout.addWidget(self.curve_plot, 2)

        self.curve_data = [0] * 40
        self.curve_plot.setYRange(0, 2)

        # 🔥 YELLOW CURVE
        self.curve_curve = self.curve_plot.plot(
            self.curve_data,
            pen=pg.mkPen(color='#FFD700', width=3)
        )

        # RIGHT SIDE
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 2)

        def create_metric(label_text):
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 18px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            right_layout.addWidget(label)
            right_layout.addWidget(bar)
            return bar

        self.cpu_progress = create_metric("CPU Usage (%)")
        self.igpu_progress = create_metric("iGPU Usage (%)")

        self.latency_label = QLabel("Inference Latency (ms)")
        self.latency_label.setAlignment(Qt.AlignCenter)
        self.latency_label.setStyleSheet("font-size: 18px;")
        right_layout.addWidget(self.latency_label)

        self.latency_value = QLabel("0")
        self.latency_value.setAlignment(Qt.AlignCenter)
        self.latency_value.setStyleSheet("font-size: 34px; font-weight: bold;")
        right_layout.addWidget(self.latency_value)

        self.samples_label = QLabel("Samples Processed")
        self.samples_label.setAlignment(Qt.AlignCenter)
        self.samples_label.setStyleSheet("font-size: 18px;")
        right_layout.addWidget(self.samples_label)

        self.samples_value = QLabel("0")
        self.samples_value.setAlignment(Qt.AlignCenter)
        self.samples_value.setStyleSheet("font-size: 34px; font-weight: bold;")
        right_layout.addWidget(self.samples_value)

        # MODE LABEL
        self.mode_label = QLabel("Mode: Standard Mode")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #00BFFF;"
        )
        right_layout.addWidget(self.mode_label)

        # -------------------------------
        # COLORED MODE BUTTONS
        # -------------------------------
        button_layout = QHBoxLayout()
        right_layout.addLayout(button_layout)

        self.btn_standard = QPushButton("Standard Mode")
        self.btn_standard.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
        """)
        self.btn_standard.clicked.connect(lambda: self.set_mode("Standard Mode"))
        button_layout.addWidget(self.btn_standard)

        self.btn_surge = QPushButton("Surge Mode")
        self.btn_surge.setStyleSheet("""
            QPushButton {
                background-color: #C62828;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
        """)
        self.btn_surge.clicked.connect(lambda: self.set_mode("Surge Mode"))
        button_layout.addWidget(self.btn_surge)

        self.btn_cpu_only = QPushButton("CPU-Only Mode")
        self.btn_cpu_only.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
        """)
        self.btn_cpu_only.clicked.connect(lambda: self.set_mode("CPU-Only Mode"))
        button_layout.addWidget(self.btn_cpu_only)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(200)

    def set_mode(self, mode):
        self.telemetry.set_mode(mode)
        self.mode_label.setText(f"Mode: {mode}")

    def update_ui(self):
        with self.telemetry.lock:
            self.cpu_progress.setValue(int(self.telemetry.cpu_usage[-1]) if self.telemetry.cpu_usage else 0)
            self.igpu_progress.setValue(int(self.telemetry.igpu_usage[-1]) if self.telemetry.igpu_usage else 0)
            self.latency_value.setText(f"{self.telemetry.latency_ms[-1]:.2f}" if self.telemetry.latency_ms else "0.00")
            self.samples_value.setText(str(self.telemetry.samples_processed))

            new_val = 1 + 0.5 * math.sin(time.time() * 5) + random.uniform(-0.05, 0.05)
            self.curve_data = self.curve_data[1:] + [new_val]
            self.curve_curve.setData(self.curve_data)

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())
