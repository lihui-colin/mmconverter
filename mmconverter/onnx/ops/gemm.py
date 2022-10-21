from ...builder import ONNXOPS as OPS
from ... import graph
from .attribute import extract_attributes


@OPS.register_module()
class Gemm:
    shortname = "gemm"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.MMNode:
        kwargs = extract_attributes(onnx_node)

        alpha = kwargs["weight_multiplier"]
        beta = kwargs["bias_multiplier"]
        trans_a = kwargs["transpose_activation"]
        trans_b = kwargs["transpose_weight"]

        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]
        if alpha == 1 and beta == 1 and not trans_a and not trans_b:
            node = graph.Linear(onnx_node.name, input_names, output_names)
            node.weight = graph.MMParameter(params[0].data)
            if len(params) > 1:
                node.bias = graph.MMParameter(params[1].data)
            out_channels, in_channels = node.weight.data.shape[:2]
            node.in_features = in_channels
            node.out_features = out_channels
        else:
            assert False
        return node
