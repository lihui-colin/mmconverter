from .. import graph
import onnx
from onnx import numpy_helper


def get_outputs_names(onnx_graph):
    output_names = [x.name for x in onnx_graph.output]
    return output_names


def get_inputs_names(onnx_graph):
    param_names = set([x.name for x in onnx_graph.initializer])
    input_names = [x.name for x in onnx_graph.input]
    input_names = [x for x in input_names if x not in param_names]
    return input_names


def Load(onnx_file, model_name) -> graph.MMGraph:
    onnx_model = onnx.load(onnx_file)
    input_names = get_inputs_names(onnx_model.graph)
    output_names = get_outputs_names(onnx_model.graph)
    opset_version = onnx_model.opset_import[0].version

    graph = graph.MMGraph(model_name)
    return graph
