import onnx
from onnx import helper, TensorProto

# Explicitly set stable opset
OPSET_VERSION = 12

input_tensor = helper.make_tensor_value_info(
    'input', TensorProto.FLOAT, [None, 40]
)

output_tensor = helper.make_tensor_value_info(
    'output', TensorProto.FLOAT, [None, 3]
)

# Weights
W = helper.make_tensor(
    name="W",
    data_type=TensorProto.FLOAT,
    dims=[40, 3],
    vals=[0.01] * (40 * 3),
)

B = helper.make_tensor(
    name="B",
    data_type=TensorProto.FLOAT,
    dims=[3],
    vals=[0.1, -0.1, 0.05],
)

# Single GEMM node (matrix multiply + bias)
node = helper.make_node(
    "Gemm",
    inputs=["input", "W", "B"],
    outputs=["output"],
)

graph = helper.make_graph(
    [node],
    "EdgePCRModel",
    [input_tensor],
    [output_tensor],
    initializer=[W, B],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", OPSET_VERSION)]
)

onnx.save(model, "edge_pcr_model.onnx")

print("Stable ONNX model created successfully.")

