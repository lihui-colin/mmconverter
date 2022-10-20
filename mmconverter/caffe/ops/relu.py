from mmconverter.graph.node import MMNode
from ...builder import CAFFEOPS as OPS
from ... import graph


@OPS.register_module()
class ReLU:
    shortname = "relu"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.MMNode:
        negative_slope = layer_param.relu_param.negative_slope
        
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom] 
        if negative_slope != 0:
            node = graph.LeakyReLU(layer_param.name, input_names, output_names)
            node.negative_slope = negative_slope
        else:
            node = graph.ReLU(layer_param.name, input_names, output_names)

        return node
