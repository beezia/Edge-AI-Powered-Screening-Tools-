from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QProgressBar, QFrame
)
from PyQt5.QtCore import QTimer, Qt
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
        self.window.setWindowTitle("Edge AI Diagnostics Demo")
        self.window.resize(900, 600)

        # Layouts
        main_layout = QHBoxLayout()
        self.window.setLayout(main_layout)

        # LEFT SIDE: Sample Processing Animation & Curve Plot
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 3)

        self.sample_label = QLabel("Sample Processing Animation")
        self.sample_label.setAlignment(Qt.AlignCenter)
        self.sample_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.sample_label)

        self.curve_plot = pg.PlotWidget(title="Amplification Curve")
        left_layout.addWidget(self.curve_plot)

        self.curve_data = [0]*40
        self.curve_plot.setYRange(0, 2)
        self.curve_curve = self.curve_plot.plot(self.curve_data, pen='y')

        # RIGHT SIDE: Telemetry Panels
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 2)

        # CPU Usage Gauge
        self.cpu_label = QLabel("CPU Usage (%)")
        self.cpu_label.setAlignment(Qt.AlignCenter)
        self.cpu_label.setStyleSheet("font-size: 20px;")
        right_layout.addWidget(self.cpu_label)

        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        right_layout.addWidget(self.cpu_progress)

        # iGPU Usage Gauge
        self.igpu_label = QLabel("iGPU Usage (%)")
        self.igpu_label.setAlignment(Qt.AlignCenter)
        self.igpu_label.setStyleSheet("font-size: 20px;")
        right_layout.addWidget(self.igpu_label)

        self.igpu_progress = QProgressBar()
        self.igpu_progress.setRange(0, 100)
        right_layout.addWidget(self.igpu_progress)

        # Inference Latency
        self.latency_label = QLabel("Inference Latency (ms)")
        self.latency_label.setAlignment(Qt.AlignCenter)
        self.latency_label.setStyleSheet("font-size: 20px;")
        right_layout.addWidget(self.latency_label)

        self.latency_value = QLabel("0")
        self.latency_value.setAlignment(Qt.AlignCenter)
        self.latency_value.setStyleSheet("font-size: 36px; font-weight: bold;")
        right_layout.addWidget(self.latency_value)

        # Samples processed
        self.samples_label = QLabel("Samples Processed")
        self.samples_label.setAlignment(Qt.AlignCenter)
        self.samples_label.setStyleSheet("font-size: 20px;")
        right_layout.addWidget(self.samples_label)

        self.samples_value = QLabel("0")
        self.samples_value.setAlignment(Qt.AlignCenter)
        self.samples_value.setStyleSheet("font-size: 36px; font-weight: bold;")
        right_layout.addWidget(self.samples_value)

        # Mode indicator
        self.mode_label = QLabel("Mode: Standard Mode")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet("font-size: 20px; font-weight: bold; color: green;")
        right_layout.addWidget(self.mode_label)

        # Buttons for modes
        button_layout = QHBoxLayout()
        right_layout.addLayout(button_layout)

        self.btn_standard = QPushButton("🟢 Standard Mode")
        self.btn_standard.clicked.connect(lambda: self.set_mode("Standard Mode"))
        button_layout.addWidget(self.btn_standard)

        self.btn_surge = QPushButton("🔴 Surge Mode")
        self.btn_surge.clicked.connect(lambda: self.set_mode("Surge Mode"))
        button_layout.addWidget(self.btn_surge)

        self.btn_cpu_only = QPushButton("🔵 CPU-Only Mode")
        self.btn_cpu_only.clicked.connect(lambda: self.set_mode("CPU-Only Mode"))
        button_layout.addWidget(self.btn_cpu_only)

        # WMI connection for GPU telemetry
     #   self.wmi_conn = wmi.WMI(namespace="root\CIMV2")

        # Timer to update UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(200)  # Update 5 times/sec

    def set_mode(self, mode):
        self.telemetry.set_mode(mode)
        self.mode_label.setText(f"Mode: {mode}")
        color = {"Standard Lab": "green", "Surge Mode": "red", "CPU-Only Mode": "blue"}.get(mode, "black")
        self.mode_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")


    def update_ui(self):
        with self.telemetry.lock:

            # CPU
            if self.telemetry.cpu_usage:
                self.cpu_progress.setValue(int(self.telemetry.cpu_usage[-1]))
            else:
                self.cpu_progress.setValue(0)

            # iGPU (from main.py telemetry)
            if self.telemetry.igpu_usage:
                self.igpu_progress.setValue(int(self.telemetry.igpu_usage[-1]))
            else:
                self.igpu_progress.setValue(0)

            # Latency
            if self.telemetry.latency_ms:
                latency = self.telemetry.latency_ms[-1]
                self.latency_value.setText(f"{latency:.1f}")
            else:
                self.latency_value.setText("0")

            # Samples
            self.samples_value.setText(str(self.telemetry.samples_processed))

            # Curve animation
            new_val = 1 + 0.5 * math.sin(time.time()*5) + random.uniform(-0.05, 0.05)
            self.curve_data = self.curve_data[1:] + [new_val]
            self.curve_curve.setData(self.curve_data)


    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())
