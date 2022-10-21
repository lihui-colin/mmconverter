from typing import OrderedDict
from ...builder import ONNXOPS as OPS
from ... import graph
from ...graph import MMParameter
import numpy as np
import torch
from .attribute import extract_attributes


@OPS.register_module()
class Conv:
    shortname = "conv"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.Conv2d:
        kwargs = extract_attributes(onnx_node)
        kernel_size = kwargs["kernel_size"] if "kernel_size" in kwargs else [1, 1]
        pads = kwargs["padding"] if "padding" in kwargs else [0, 0]
        strides = kwargs["stride"] if "stride" in kwargs else [1, 1]
        dilation = kwargs["dilation"] if "dilation" in kwargs else [1, 1]
        groups = kwargs["groups"] if "groups" in kwargs else 1

        if isinstance(pads, torch.nn.ConstantPad2d) or isinstance(
            pads, torch.nn.ConstantPad3d
        ):
            pads = [0, 0]
            assert False

        weight = MMParameter(params[0].data)
        bias = MMParameter(params[1].data) if len(params) > 1 else None
        output_channels, input_channels = weight.data.shape[:2]
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]

        node = graph.Conv2d(onnx_node.name, input_names, output_names)
        node.in_channels = input_channels
        node.out_channels = output_channels
        node.kernel_size = kernel_size
        node.stride = strides
        node.padding = pads
        node.dilation = dilation
        node.groups = groups
        node.padding_mode = "zeros"
        node.weight = weight
        node.bias = bias
        return node
