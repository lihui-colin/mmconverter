
from typing import OrderedDict
from ...builder import CAFFEOPS as OPS
from ... import graph
import numpy as np
import torch


@OPS.register_module()
class Softmax:
    shortname = "softmax"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.Softmax:
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom]
        node = graph.Softmax(layer_param.name, input_names, output_names)
        node.dim = layer_param.softmax_param.axis
        return node
