from mmconverter.graph.node import MMNode
from ...builder import ONNXOPS as OPS
from ... import graph
from .attribute import extract_attributes


@OPS.register_module()
class Relu:
    shortname = "relu"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.MMNode:
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]
        node = graph.ReLU(onnx_node.name, input_names, output_names)
        return node


@OPS.register_module()
class LeakyRelu:
    shortname = "leakyrelu"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.MMNode:
        kwargs = extract_attributes(onnx_node)
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]
        node = graph.LeakyReLU(onnx_node.name, input_names, output_names)
        node.negative_slope = kwargs["negative_slope"]
        return node
