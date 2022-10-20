from ...builder import CAFFEOPS as OPS
from ... import graph


@OPS.register_module()
class Eltwise:
    shortname = "eltwise"
    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.MMNode:
        # enum EltwiseOp {
        #     PROD = 0;
        #     SUM = 1;
        #     MAX = 2;
        # }
        
        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom] 
        
        if layer_param.eltwise_param.operation == 1:
            node = graph.Add(layer_param.name, input_names, output_names)
        elif layer_param.eltwise_param.operation == 0:
            node = graph.Mul(layer_param.name, input_names, output_names)
        elif layer_param.eltwise_param.operation == 2:
            assert False
        else:
            assert False
        return node
