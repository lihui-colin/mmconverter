from ...builder import ONNXOPS as OPS
from ... import graph
from .attribute import extract_attributes
import torch


@OPS.register_module()
class MaxPool:
    shortname = "maxpool"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.MMNode:
        kwargs = extract_attributes(onnx_node)
        kernel_size = kwargs["kernel_size"] if "kernel_size" in kwargs else [1, 1]
        pads = (
            kwargs["pads"] if "pads" in kwargs else [0, 0]
        )  # onnx pad支持四个方向，可能需要添加Pad节点
        strides = kwargs["stride"] if "stride" in kwargs else [1, 1]
        ceil_mode = kwargs["ceil_mode"] if "ceil_mode" in kwargs else False
        
        if isinstance(pads, torch.nn.ConstantPad2d) or isinstance(
            pads, torch.nn.ConstantPad3d
        ):
            pads = [0, 0]
            assert False

        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]

        node = graph.MaxPool2d(onnx_node.name, input_names, output_names)
        node.kernel_size = kernel_size
        node.stride = strides
        node.padding = pads
        node.ceil_mode = ceil_mode

        return node


@OPS.register_module()
class AveragePool:
    shortname = "avgpool"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.MMNode:
        kwargs = extract_attributes(onnx_node)
        kernel_size = kwargs["kernel_size"] if "kernel_size" in kwargs else [2, 2]
        pads = (
            kwargs["padding"] if "padding" in kwargs else [0, 0]
        )  # onnx pad支持四个方向，可能需要添加Pad节点
        strides = kwargs["stride"] if "stride" in kwargs else [1, 1]
        if isinstance(pads, torch.nn.ConstantPad2d) or isinstance(
            pads, torch.nn.ConstantPad3d
        ):
            pads = [0, 0]
            assert False
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]

        node = graph.AvgPool2d(onnx_node.name, input_names, output_names)
        node.kernel_size = kernel_size
        node.stride = strides
        node.padding = pads

        return node
