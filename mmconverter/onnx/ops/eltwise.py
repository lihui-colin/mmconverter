from ...builder import ONNXOPS as OPS
from ... import graph


@OPS.register_module()
class Add:
    shortname = "add"

    def __init__(self) -> None:
        pass

    def __call__(self, onnx_node, params) -> graph.MMNode:
        output_names = onnx_node.output
        input_names = onnx_node.input
        node = graph.Add(onnx_node.name, input_names, output_names)
        return node


# @OPS.register_module()
# class Mul:
#     shortname = "mul"
#     def __init__(self) -> None:
#         pass

#     def __call__(self, onnx_node, params) -> graph.MMNode:

#         output_names = onnx_node.output
#         input_names = onnx_node.input

#         if layer_param.eltwise_param.operation == 1:
#             node = graph.Add(onnx_node.name, input_names, output_names)
#         elif layer_param.eltwise_param.operation == 0:
#             node = graph.Mul(layer_param.name, input_names, output_names)
#         elif layer_param.eltwise_param.operation == 2:
#             assert False
#         else:
#             assert False
#         return node
