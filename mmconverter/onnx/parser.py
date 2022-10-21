from asyncio.log import logger
from ..graph import MMGraph
import onnx
from onnx import numpy_helper
from ..builder import ONNXOPS as OPS
from . import ops
from loguru import logger
from .blob import Blob
from tqdm import tqdm

from mmconverter import graph


def get_outputs_names(onnx_graph):
    output_names = [x.name for x in onnx_graph.output]
    return output_names


def get_inputs_names(onnx_graph):
    param_names = set([x.name for x in onnx_graph.initializer])
    input_names = [x.name for x in onnx_graph.input]
    input_names = [x for x in input_names if x not in param_names]
    return input_names


def Load(onnx_file, model_name) -> MMGraph:
    onnx_model = onnx.load(onnx_file)
    input_names = get_inputs_names(onnx_model.graph)
    output_names = get_outputs_names(onnx_model.graph)
    opset_version = onnx_model.opset_import[0].version
    onnx_graph = onnx_model.graph

    # 检查是否存在不支持的OP
    unsupport_ops = []
    for node in onnx_graph.node:
        cls_obj = OPS.get(node.op_type)
        if cls_obj is None and node.op_type not in unsupport_ops:
            unsupport_ops.append(node.op_type)

    if len(unsupport_ops):
        logger.error("Unsupport OP:")
        for op in unsupport_ops:
            logger.error(f"  {op}")
        return None

    logger.info(f"generating graph")
    mm_graph = MMGraph(model_name)
    weights = {tensor.name: tensor for tensor in onnx_graph.initializer}

    mm_nodes = []
    for i, i_name in enumerate(input_names):
        i_name = f"{i_name}"
        input_node = graph.Input(f"input_{i}", [i_name], [i_name])
        mm_nodes.append(input_node)

    for onnx_node in tqdm(onnx_graph.node):
        params = [
            weights[par_name] for par_name in onnx_node.input if par_name in weights
        ]
        params = [Blob(blob) for blob in params]
        cls_obj = OPS.get(onnx_node.op_type)
        node = cls_obj()(onnx_node, params)
        if isinstance(node, list):
            mm_nodes += node
        else:
            mm_nodes.append(node)

    for node in mm_nodes:
        node.name = f"mm_{node.name}"
        node.input_names = [f"var_{x}" for x in node.input_names]
        node.output_names = [f"var_{x}" for x in node.output_names]
        mm_graph.addNode(node)
    logger.info(f"optimize graph")
    mm_graph.resort_nodes()
    return mm_graph
