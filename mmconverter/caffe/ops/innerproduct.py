from typing import OrderedDict
from ...builder import CAFFEOPS as OPS
from ... import graph
import numpy as np
import torch


@OPS.register_module()
class InnerProduct:
    shortname = "fc"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params):
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom]
        node = graph.Linear(layer_param.name, input_names, output_names)
        bias = layer_param.inner_product_param.bias_term 
        
        node.in_features = params[0].data.shape[1]
        node.out_features = layer_param.inner_product_param.num_output
        node.weight = graph.MMParameter(params[0].data)
        if bias:
            node.bias = graph.MMParameter(params[1].data)
            
        reshape_node = graph.Reshape(input_names[0], input_names, input_names)
        reshape_node.shape = [-1, node.in_features]
        
        return [reshape_node, node]
