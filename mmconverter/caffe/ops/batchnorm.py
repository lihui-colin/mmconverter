from typing import OrderedDict
from ...builder import CAFFEOPS as OPS
from ... import graph
import numpy as np
import torch


@OPS.register_module()
class BatchNorm:
    shortname = "bn"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.BatchNorm2d:
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom]
        node = graph.BatchNorm2d(layer_param.name, input_names, output_names)

        node.eps = layer_param.batch_norm_param.eps
        node.momentum = layer_param.batch_norm_param.moving_average_fraction
        node.affine = True
        node.track_running_stats = True

        node.running_mean = graph.MMParameter(params[0].data / params[2].data)
        node.running_var = graph.MMParameter(params[1].data / params[2].data)
        node.num_features = node.running_mean.data.shape[0]
        node.weight = graph.MMParameter(torch.ones(node.num_features))
        node.bias = graph.MMParameter(torch.zeros(node.num_features))
        return node


@OPS.register_module()
class Scale:
    shortname = "scale"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.Scale:
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom]
        node = graph.Scale(layer_param.name, input_names, output_names)
        node.weight = graph.MMParameter(params[0].data)
        node.bias = graph.MMParameter(params[1].data)
        return node
