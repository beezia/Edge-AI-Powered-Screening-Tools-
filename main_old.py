import sys
import time
import threading
import numpy as np
import psutil
import onnxruntime as ort
from ui import EdgeAIDemoUI

CYCLES = 40

class TelemetryData:
    def __init__(self):
        self.lock = threading.Lock()
        self.cpu_usage = []
        self.igpu_usage = []
        self.latency_ms = []
        self.samples_processed = 0
        self.mode = "Standard Lab"
        self.batch_size = 1
        self.use_igpu = True

    def update(self, cpu, igpu, latency, batch):
        with self.lock:
            self.cpu_usage.append(cpu)
            self.igpu_usage.append(igpu)
            self.latency_ms.append(latency)
            self.samples_processed += batch
            # Keep telemetry lists trimmed
            self.cpu_usage = self.cpu_usage[-100:]
            self.igpu_usage = self.igpu_usage[-100:]
            self.latency_ms = self.latency_ms[-100:]

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            if mode == "Standard Lab":
                self.batch_size = 1
                self.use_igpu = True
            elif mode == "Surge Mode":
                self.batch_size = 4
                self.use_igpu = True
            elif mode == "CPU-Only Mode":
                self.batch_size = 1
                self.use_igpu = False

def generate_curve():
    x = np.arange(CYCLES)
    baseline = np.random.normal(0.05, 0.02, CYCLES)
    signal = 1 / (1 + np.exp(-(x - 25)/2)) if np.random.rand() > 0.5 else np.zeros(CYCLES)
    noise = np.random.normal(0, 0.02, CYCLES)
    return (baseline + signal + noise).astype(np.float32)

def preprocess(batch):
    batch = (batch - batch.min(axis=1, keepdims=True)) / (batch.max(axis=1, keepdims=True) + 1e-6)
    return batch.astype(np.float32)

def create_session(use_igpu: bool):
    try:
        if use_igpu:
            print("Attempting DirectML (iGPU)...")
            session = ort.InferenceSession(
                "model/edge_pcr_model.onnx",
                providers=["DmlExecutionProvider"]
            )
            print("DirectML session created successfully.")
        else:
            print("Using CPU Execution Provider...")
            session = ort.InferenceSession(
                "model/edge_pcr_model.onnx",
                providers=["CPUExecutionProvider"]
            )
            print("CPU session created successfully.")

        return session

    except Exception as e:
        print("DirectML failed. Falling back to CPU.")
        print("Error:", e)

        return ort.InferenceSession(
            "model/edge_pcr_model.onnx",
            providers=["CPUExecutionProvider"]
        )


def run_inference_loop(telemetry: TelemetryData):

    # Create session ONCE
    session = create_session(telemetry.use_igpu)


    print("Inference session initialized successfully")

    # Prime CPU measurement
    psutil.cpu_percent(interval=None)

    while True:
        try:
            
            # If execution mode changed, recreate session
                if telemetry.use_igpu and "DmlExecutionProvider" not in session.get_providers():
                    session = create_session(True)

                if not telemetry.use_igpu and "CPUExecutionProvider" not in session.get_providers():
                    session = create_session(False)

            batch_size = telemetry.batch_size

            # Generate synthetic batch
            batch = np.array([generate_curve() for _ in range(batch_size)])
            batch = preprocess(batch)

            start = time.time()

            outputs = session.run(
                ["output"],
                {"input": batch}
            )

            latency = (time.time() - start) * 1000

            cpu_percent = psutil.cpu_percent(interval=None)

            telemetry.update(
                cpu_percent,
                0,  # GPU placeholder
                latency,
                batch_size
            )

            print(f"Latency: {latency:.2f} ms | CPU: {cpu_percent}%")

            time.sleep(0.1)

        except Exception as e:
            print("INFERENCE LOOP ERROR:", e)
            time.sleep(1)


def main():
    telemetry = TelemetryData()

    # Start inference loop in background thread
    thread = threading.Thread(target=run_inference_loop, args=(telemetry,), daemon=True)
    thread.start()

    # Start PyQt UI and pass telemetry object
    app = EdgeAIDemoUI(telemetry)
    app.run()

if __name__ == "__main__":
    main()
