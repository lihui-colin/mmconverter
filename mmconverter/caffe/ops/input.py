from ...builder import CAFFEOPS as OPS
from ... import graph


@OPS.register_module()
class Input:
    shortname = "input"

    def __init__(self) -> None:
        pass

    def create_forward(self):
        return None

    def __call__(self, layer_param, params) -> graph.MMNode:
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom]
        return graph.Input(layer_param.name, input_names, output_names)
