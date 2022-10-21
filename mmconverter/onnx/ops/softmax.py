
from ...builder import ONNXOPS as OPS
from ... import graph
from .attribute import extract_attributes

@OPS.register_module()
class Softmax:
    shortname = "softmax"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.Softmax:
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]] 
        kwargs = extract_attributes(onnx_node) 
        node = graph.Softmax(onnx_node.name, input_names, output_names)
        node.dim = kwargs['dim']
        return node
