import sys
import time
import threading
import numpy as np
import onnxruntime as ort
import psutil
import wmi

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --------------------------------
# WMI Setup for GPU utilization
# --------------------------------
w = wmi.WMI(namespace="root\\CIMV2")

def get_igpu_usage():
    # Approximate Intel GPU usage via WMI query of GPU engine
    gpu_loads = []
    try:
        for gpu in w.Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine():
            # Filter for Intel iGPU by checking Name contains "Intel"
            if "intel" in gpu.Name.lower():
                gpu_loads.append(int(gpu.UtilizationPercentage))
    except Exception:
        return 0
    if gpu_loads:
        return max(gpu_loads)
    return 0

# --------------------------------
# ONNX Inference Setup
# --------------------------------

NUM_SAMPLES = 1
CYCLES = 40
BATCH_SIZE = 1
USE_IGPU = True  # Set False for CPU; True if you want to enable DML provider

MODEL_PATH = "edge_pcr_model.onnx"

providers = ["CPUExecutionProvider"]
if USE_IGPU:
    providers = ["DmlExecutionProvider"]

session = ort.InferenceSession(MODEL_PATH, providers=providers)

def generate_curve():
    x = np.arange(CYCLES)
    baseline = np.random.normal(0.05, 0.02, CYCLES)
    if np.random.rand() > 0.5:
        signal = 1 / (1 + np.exp(-(x - 25) / 2))
    else:
        signal = np.zeros(CYCLES)
    noise = np.random.normal(0, 0.02, CYCLES)
    return (baseline + signal + noise).astype(np.float32)

def generate_batch():
    return np.array([generate_curve() for _ in range(BATCH_SIZE)])

def preprocess(batch):
    batch = (batch - batch.min(axis=1, keepdims=True)) / (batch.max(axis=1, keepdims=True) + 1e-6)
    return batch.astype(np.float32)

# --------------------------------
# PyQt UI Setup
# --------------------------------

class TelemetryUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Edge Compute Telemetry")
        self.setGeometry(100, 100, 800, 600)

        self.cpu_data = []
        self.igpu_data = []
        self.latency_data = []

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Labels
        self.cpu_label = QLabel("CPU Usage: 0 %")
        self.gpu_label = QLabel("iGPU Usage: 0 %")
        self.latency_label = QLabel("Inference Latency: 0 ms")

        for lbl in (self.cpu_label, self.gpu_label, self.latency_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
            self.layout.addWidget(lbl)

        # Matplotlib Figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax_cpu = self.figure.add_subplot(311)
        self.ax_gpu = self.figure.add_subplot(312)
        self.ax_latency = self.figure.add_subplot(313)

        self.ax_cpu.set_title("CPU Usage (%)")
        self.ax_gpu.set_title("Intel iGPU Usage (%)")
        self.ax_latency.set_title("Inference Latency (ms)")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second

    def update_plot(self):
        self.ax_cpu.cla()
        self.ax_gpu.cla()
        self.ax_latency.cla()

        self.ax_cpu.plot(self.cpu_data[-30:], color='blue')
        self.ax_gpu.plot(self.igpu_data[-30:], color='green')
        self.ax_latency.plot(self.latency_data[-30:], color='red')

        self.ax_cpu.set_ylim(0, 20)
        self.ax_gpu.set_ylim(0, 20)
        # Latency auto-scale
        if self.latency_data:
            max_lat = max(self.latency_data[-30:])
            self.ax_latency.set_ylim(0, max(20, max_lat*1.2))
        else:
            self.ax_latency.set_ylim(0, 20)

        self.ax_cpu.set_title("CPU Usage (%)")
        self.ax_gpu.set_title("Intel iGPU Usage (%)")
        self.ax_latency.set_title("Inference Latency (ms)")

        self.canvas.draw()

    def update_telemetry(self, cpu, gpu, latency):
        self.cpu_label.setText(f"CPU Usage: {cpu:.1f} %")
        self.gpu_label.setText(f"iGPU Usage: {gpu:.1f} %")
        self.latency_label.setText(f"Inference Latency: {latency:.2f} ms")

        self.cpu_data.append(cpu)
        self.igpu_data.append(gpu)
        self.latency_data.append(latency)

# --------------------------------
# Background Inference Thread
# --------------------------------

def inference_loop(ui: TelemetryUI):
    while True:
        batch = generate_batch()
        batch = preprocess(batch)

        start = time.time()
        outputs = session.run(["output"], {"input": batch})
        latency = (time.time() - start) * 1000

        cpu = psutil.cpu_percent(interval=None)
        gpu = get_igpu_usage()

        # Send telemetry to UI
        ui.update_telemetry(cpu, gpu, latency)

        time.sleep(0.5)  # Adjust frequency

# --------------------------------
# Main Entry Point
# --------------------------------

def main():
    app = QApplication(sys.argv)

    ui = TelemetryUI()
    ui.show()

    # Run inference in a separate thread
    thread = threading.Thread(target=inference_loop, args=(ui,), daemon=True)
    thread.start()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
