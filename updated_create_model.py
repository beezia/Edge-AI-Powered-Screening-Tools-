import onnx
from onnx import helper, TensorProto
import numpy as np

# Stable opset for DirectML
OPSET_VERSION = 12

# -----------------------------
# Model Dimensions
# -----------------------------
INPUT_FEATURES = 40
HIDDEN_UNITS = 1024   # <-- Heavy compute layer

# -----------------------------
# Dynamic Batch Input
# -----------------------------
input_tensor = helper.make_tensor_value_info(
    'input',
    TensorProto.FLOAT,
    ['batch_size', INPUT_FEATURES]   # Dynamic batch
)

output_tensor = helper.make_tensor_value_info(
    'output',
    TensorProto.FLOAT,
    ['batch_size', HIDDEN_UNITS]
)

# -----------------------------
# Initialize Weights
# -----------------------------
W_data = np.random.randn(INPUT_FEATURES, HIDDEN_UNITS).astype(np.float32) * 0.01
B_data = np.random.randn(HIDDEN_UNITS).astype(np.float32) * 0.01

W = helper.make_tensor(
    name="W",
    data_type=TensorProto.FLOAT,
    dims=[INPUT_FEATURES, HIDDEN_UNITS],
    vals=W_data.flatten().tolist()
)

B = helper.make_tensor(
    name="B",
    data_type=TensorProto.FLOAT,
    dims=[HIDDEN_UNITS],
    vals=B_data.tolist()
)

# -----------------------------
# GEMM Node (Matrix Multiply + Bias)
# -----------------------------
node = helper.make_node(
    "Gemm",
    inputs=["input", "W", "B"],
    outputs=["output"],
    alpha=1.0,
    beta=1.0,
    transB=0
)

# -----------------------------
# Build Graph
# -----------------------------
graph = helper.make_graph(
    [node],
    "EdgePCRDynamicModel",
    [input_tensor],
    [output_tensor],
    initializer=[W, B],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", OPSET_VERSION)]
)

# Make IR version compatible with most ONNX Runtime builds
model.ir_version = 9

# -----------------------------
# Save Model
# -----------------------------
onnx.save(model, "edge_pcr_model.onnx")

print("✅ Dynamic 1024 ONNX model created successfully.")
