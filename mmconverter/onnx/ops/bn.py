from typing import OrderedDict
from ...builder import ONNXOPS as OPS
from ... import graph
import numpy as np
import torch
from .attribute import extract_attributes


@OPS.register_module()
class BatchNormalization:
    shortname = "bn"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.BatchNorm2d:
        output_names = onnx_node.output
        input_names = [onnx_node.input[0]]
        kwargs = extract_attributes(onnx_node)

        node = graph.BatchNorm2d(onnx_node.name, input_names, output_names)

        node.eps = kwargs["eps"]
        node.momentum = kwargs["momentum"]
        node.affine = True
        node.track_running_stats = True

        node.weight = graph.MMParameter(params[0].data)
        node.bias = graph.MMParameter(params[1].data)
        node.running_mean = graph.MMParameter(params[2].data)
        node.running_var = graph.MMParameter(params[3].data)
        node.num_features = node.running_mean.data.shape[0]
        return node
