from ...builder import ONNXOPS as OPS
from ... import graph
import torch
from .attribute import extract_attributes

@OPS.register_module()
class Reshape:
    shortname = "reshape"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params):
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]
        
        node = graph.Reshape(onnx_node.name, input_names, output_names)
        node.shape = params[0].data.to(torch.int).numpy().tolist()
        return node
