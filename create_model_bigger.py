import onnx
from onnx import helper, TensorProto

OPSET_VERSION = 12

# Use symbolic dimension instead of None
input_tensor = helper.make_tensor_value_info(
    'input',
    TensorProto.FLOAT,
    ['batch_size', 40]
)

output_tensor = helper.make_tensor_value_info(
    'output',
    TensorProto.FLOAT,
    ['batch_size', 512]  # increased for GPU load
)

# Bigger weight matrix for visible GPU load
W = helper.make_tensor(
    name="W",
    data_type=TensorProto.FLOAT,
    dims=[40, 512],
    vals=[0.01] * (40 * 512),
)

B = helper.make_tensor(
    name="B",
    data_type=TensorProto.FLOAT,
    dims=[512],
    vals=[0.1] * 512,
)

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

onnx.save(model, "model/edge_pcr_model.onnx")

print("Dynamic batch ONNX model created successfully.")

