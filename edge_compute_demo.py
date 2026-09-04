import numpy as np
import onnxruntime as ort
import time
import psutil
import matplotlib.pyplot as plt
from threading import Thread
import os

# ----------------------------------------
# Configuration
# ----------------------------------------

NUM_SAMPLES = 96
CYCLES = 40
BATCH_SIZE = 1
USE_IGPU = True  # Toggle between CPU and iGPU

MODEL_PATH = "edge_pcr_model.onnx"

# ----------------------------------------
# Synthetic PCR Curve Generator (CPU)
# ----------------------------------------

def generate_curve():
    x = np.arange(CYCLES)
    baseline = np.random.normal(0.05, 0.02, CYCLES)

    if np.random.rand() > 0.5:
        signal = 1 / (1 + np.exp(-(x - 25)/2))
    else:
        signal = np.zeros(CYCLES)

    noise = np.random.normal(0, 0.02, CYCLES)
    return (baseline + signal + noise).astype(np.float32)

def generate_batch():
    return np.array([generate_curve() for _ in range(BATCH_SIZE)])

# ----------------------------------------
# CPU Signal Filtering
# ----------------------------------------

def preprocess(batch):
    batch = (batch - batch.min(axis=1, keepdims=True)) / \
            (batch.max(axis=1, keepdims=True) + 1e-6)
    return batch.astype(np.float32)

# ----------------------------------------
# ONNX Runtime Setup
# ----------------------------------------

providers = ["CPUExecutionProvider"]
if USE_IGPU:
    providers = ["DmlExecutionProvider"]

session = ort.InferenceSession(MODEL_PATH, providers=providers)

# ----------------------------------------
# Telemetry
# ----------------------------------------

cpu_usage = []
inference_times = []

# ----------------------------------------
# Visualization Setup
# ----------------------------------------

plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

def update_plot():
    axs[0].cla()
    axs[1].cla()

    axs[0].plot(cpu_usage[-30:])
    axs[0].set_title("CPU Usage (%)")

    axs[1].plot(inference_times[-30:])
    axs[1].set_title("Inference Latency (ms)")

    plt.tight_layout()
    plt.pause(0.1)

# ----------------------------------------
# Main Loop
# ----------------------------------------

print("Starting Edge Compute Demo...")
print("Using iGPU:", USE_IGPU)

while True:
    batch = generate_batch()               # CPU generation
    batch = preprocess(batch)              # CPU filtering

    start = time.time()
    outputs = session.run(
        ["output"],
        {"input": batch}
    )
    latency = (time.time() - start) * 1000

    inference_times.append(latency)
    cpu_usage.append(psutil.cpu_percent())

    print(f"Inference Latency: {latency:.2f} ms")

    update_plot()
