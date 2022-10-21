from .proto import caffe_upsample_pb2 as caffe_pb2
from google.protobuf import text_format
from ..builder import CAFFEOPS as OPS
from .ops import *
from ..graph import MMGraph
from ..graph import ops
from loguru import logger
from .blob import Blob
from tqdm import tqdm


def LoadCaffeModel(net_path, model_path):
    # read prototxt
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), net)
    # read caffemodel
    model = caffe_pb2.NetParameter()
    f = open(model_path, "rb")
    model.ParseFromString(f.read())
    f.close()
    return net, model


# 获取网络层
def GetNetLayerCaffe(net):
    if len(net.layer) == 0 and len(net.layers) != 0:
        return net.layers
    elif len(net.layer) != 0 and len(net.layers) == 0:
        return net.layer
    else:
        print("prototxt layer error")
        return -1


# 获取参数层
def GetNetModelCaffe(model):
    if len(model.layer) == 0 and len(model.layers) != 0:
        return model.layers
    elif len(model.layer) != 0 and len(model.layers) == 0:
        return model.layer
    else:
        print("caffemodel layer error")
        return -1


# 获取输入层
def GetInputLayers(net):
    input_layers = []
    for layer in net:
        if layer.type == "Input":
            input_layers.append(layer)
    return input_layers


def IsVarName(str):
    return str[0].isalpha()


def MergeBN(graph):
    mask = [False] * len(graph.nodes)
    filter_nodes = []
    for i in range(len(graph.nodes)):
        if mask[i]:
            continue
        mask[i] = True
        filter_nodes.append(graph.nodes[i])
        if (
            isinstance(graph.nodes[i], ops.BatchNorm2d)
            and ((i + 1) < len(graph.nodes))
            and isinstance(graph.nodes[i + 1], ops.Scale)
        ):
            bn_node = graph.nodes[i]
            scale_node = graph.nodes[i + 1]
            bn_node.weight = scale_node.weight
            bn_node.bias = scale_node.bias
            bn_node.output_names = scale_node.output_names
            mask[i + 1] = True
    graph.nodes = filter_nodes


def Load(caffe_proto_file, caffe_model_file, model_name):
    logger.info(f"load caffe model: {caffe_proto_file} {caffe_model_file}")
    graph, params = LoadCaffeModel(caffe_proto_file, caffe_model_file)
    netLayerCaffe = GetNetLayerCaffe(graph)
    netModelCaffe = GetNetModelCaffe(params)

    # 检查是否存在不支持的OP
    unsupport_ops = []
    for layer in netLayerCaffe:
        cls_obj = OPS.get(layer.type)
        if cls_obj is None and layer.type not in unsupport_ops:
            unsupport_ops.append(layer.type)

    if len(unsupport_ops):
        logger.error("Unsupport OP:")
        for op in unsupport_ops:
            logger.error(f"  {op}")
        return None

    top_names = {}
    for layer in netLayerCaffe:
        cls_obj = OPS.get(layer.type)
        shortname = cls_obj.shortname
        if not IsVarName(layer.name):
            # 层名称不符合python规范，修改层名称
            if not IsVarName(layer.name):
                new_name = f"{shortname}_{layer.name}"
                for layer_param in netModelCaffe:
                    if layer.name == layer_param.name:
                        layer_param.name = new_name
                        layer.name = new_name
                        break

        # 层输出变量名称不符合python规范，则需要修改
        for out_var in layer.top:
            out_var_name = out_var
            if not IsVarName(out_var_name):
                out_var_name = f"{shortname}_{out_var_name}"
            # assert out_var_name not in top_names, out_var_name
            top_names[out_var] = out_var_name

            for i in range(len(layer.top)):
                layer.top[i] = top_names[layer.top[i]]

            for i in range(len(layer.bottom)):
                layer.bottom[i] = top_names[layer.bottom[i]]

    logger.info(f"generating graph")
    graph = MMGraph(model_name)
    for layer in tqdm(netLayerCaffe):
        params = []
        for layer_param in netModelCaffe:
            if layer.name == layer_param.name:
                params = [Blob(blob) for blob in layer_param.blobs]
        cls_obj = OPS.get(layer.type)
        node = cls_obj()(layer, params)
        if isinstance(node, list):
            for s_node in node:
                graph.addNode(s_node)
        else:
            graph.addNode(node)
    
    logger.info(f"optimize graph")
    MergeBN(graph) 
    graph.resort_nodes()
    return graph
