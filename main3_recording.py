import sys
import time
import threading
import numpy as np
import psutil
import onnxruntime as ort
from ui import EdgeAIDemoUI
import subprocess
import os
import re

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = "model/edge_pcr_model.onnx"
CYCLES = 40

# --------------------------------------------------
# Telemetry Class
# --------------------------------------------------
class TelemetryData:
    def __init__(self):
        self.lock = threading.Lock()
        self.cpu_usage = []
        self.igpu_usage = []
        self.latency_ms = []
        self.samples_processed = 0

        self.mode = "Standard Mode"
        self.batch_size = 1
        self.use_igpu = True

    def update(self, cpu, igpu, latency, batch):
        with self.lock:
            self.cpu_usage.append(cpu)
            self.igpu_usage.append(igpu)
            self.latency_ms.append(latency)
            self.samples_processed += batch

            # Keep last 100 values only
            self.cpu_usage = self.cpu_usage[-100:]
            self.igpu_usage = self.igpu_usage[-100:]
            self.latency_ms = self.latency_ms[-100:]

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            if mode == "Standard Mode":
                self.batch_size = 1
                self.use_igpu = True
            elif mode == "Surge Mode":
                self.batch_size = 128
                self.use_igpu = True
            elif mode == "CPU-Only Mode":
                self.batch_size = 1
                self.use_igpu = False

# --------------------------------------------------
# GPU Telemetry
# --------------------------------------------------
def get_igpu_usage():
    """
    Returns approximate integrated GPU usage % using typeperf
    """
    try:
        pid = os.getpid()
        cmd = [
            "typeperf",
            "\\GPU Engine(*)\\Utilization Percentage",
            "-sc", "1"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        total = 0.0
        count = 0
        for line in lines:
            if f"pid_{pid}" in line:
                numbers = re.findall(r"\d+\.\d+", line)
                if numbers:
                    total += float(numbers[-1])
                    count += 1
        if count > 0:
            return total
        return 0.0
    except Exception as e:
        print("GPU telemetry error:", e)
        return 0.0

# --------------------------------------------------
# Model Session Creation
# --------------------------------------------------
def create_session(use_igpu: bool):
    try:
        if use_igpu:
            print("Attempting DirectML (iGPU)...")
            session = ort.InferenceSession(
                MODEL_PATH,
                providers=["DmlExecutionProvider"]
            )
            print("DirectML session created successfully.")
        else:
            print("Using CPU Execution Provider...")
            session = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
            print("CPU session created successfully.")
        print("Active Providers:", session.get_providers())
        return session
    except Exception as e:
        print("DirectML failed. Falling back to CPU.")
        print("Error:", e)
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        print("CPU fallback session created.")
        return session

# --------------------------------------------------
# Synthetic Data Generation
# --------------------------------------------------
def generate_curve():
    x = np.arange(CYCLES)
    baseline = np.random.normal(0.05, 0.02, CYCLES)
    signal = 1 / (1 + np.exp(-(x - 25) / 2)) if np.random.rand() > 0.5 else np.zeros(CYCLES)
    noise = np.random.normal(0, 0.02, CYCLES)
    return (baseline + signal + noise).astype(np.float32)

def preprocess(batch):
    batch = (batch - batch.min(axis=1, keepdims=True)) / (
        batch.max(axis=1, keepdims=True) + 1e-6
    )
    return batch.astype(np.float32)

# --------------------------------------------------
# Inference Loop
# --------------------------------------------------
def run_inference_loop(telemetry: TelemetryData):
    session = None
    last_batch_size = None
    last_use_igpu = None

    psutil.cpu_percent(interval=None)
    print("Inference loop started...")
    smoothed_igpu = 0.0  # Start at 0%
    alpha = 0.2           # Smoothing factor (0 < alpha <= 1)


    while True:
        try:
            batch_size = telemetry.batch_size
            use_igpu = telemetry.use_igpu

            if session is None or batch_size != last_batch_size or use_igpu != last_use_igpu:
                print("Rebuilding ONNX session...")
                print(f"  Batch size: {batch_size}")
                print(f"  Using iGPU: {use_igpu}")
                session = create_session(use_igpu)
                last_batch_size = batch_size
                last_use_igpu = use_igpu
                print("Session ready.\n")

            # Generate synthetic batch
            batch = np.array([generate_curve() for _ in range(batch_size)], dtype=np.float32)
            batch = preprocess(batch)

            # Run inference multiple times to simulate GPU load
            start = time.time()
            repeat = 200 # was 200
            for _ in range(repeat):
                session.run(None, {"input": batch})
           # time.sleep(0.05) # give main thread breathing room
            latency = (time.time() - start) * 1000 / repeat
            cpu_percent = psutil.cpu_percent(interval=None)

            # -----------------------
            # iGPU telemetry + fake multiplier + smoothing
            # -----------------------
            try:
                igpu_real = get_igpu_usage()
            except:
                igpu_real = 0.0

            # Apply trade-show multiplier
            if telemetry.mode == "Standard Lab":
                igpu_percent = min(100.0, igpu_real + 15)
            elif telemetry.mode == "Surge Mode":
                igpu_percent = min(100.0, igpu_real*8 + 20)
            elif telemetry.mode == "CPU-Only Mode":
                igpu_percent = 0.0
            else:
                igpu_percent = igpu_real

            #telemetry.update(cpu_percent, igpu_percent, latency, batch_size)
            
            # Smooth the iGPU value
            smoothed_igpu = smoothed_igpu + alpha * (igpu_percent - smoothed_igpu)

            telemetry.update(cpu_percent, smoothed_igpu, latency, batch_size)


            print(
                f"Latency: {latency:.2f} ms | "
                f"CPU: {cpu_percent:.1f}% | "
                f"iGPU: {igpu_percent:.1f}% | "
                f"Batch: {batch_size}"
            )

            time.sleep(0.1)

        except Exception as e:
            print("INFERENCE LOOP ERROR:", e)
            time.sleep(1)

# --------------------------------------------------
# Main Entry
# --------------------------------------------------
def main():
    telemetry = TelemetryData()
    print("Available Providers:", ort.get_available_providers())

    thread = threading.Thread(target=run_inference_loop, args=(telemetry,), daemon=True)
    thread.start()

    print("Starting UI...")
    app = EdgeAIDemoUI(telemetry)
    app.run()

if __name__ == "__main__":
    main()
