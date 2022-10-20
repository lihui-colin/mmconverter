import imp
from ...builder import CAFFEOPS as OPS
from ... import graph
from loguru import logger


@OPS.register_module()
class Pooling:
    shortname = "pooling"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.MMNode:
        pool_value = layer_param.pooling_param.pool
        global_value = layer_param.pooling_param.global_pooling

        kernel_size = layer_param.pooling_param.kernel_size
        pads = layer_param.pooling_param.pad
        strides = layer_param.pooling_param.stride

        if (
            layer_param.pooling_param.kernel_h != 0
            and layer_param.pooling_param.kernel_w != 0
        ):
            kernel_size = [
                layer_param.pooling_param.kernel_h,
                layer_param.pooling_param.kernel_w,
            ]

        # pass pad
        if (
            layer_param.pooling_param.pad_h != 0
            and layer_param.pooling_param.pad_w != 0
        ):
            pads = [
                layer_param.pooling_param.pad_h,
                layer_param.pooling_param.pad_w,
            ]

        # # 由于 caffe 与 onnx 的 pad 的计算的原因，将 pad 属性，单独创建一个节点
        # pads = [0, 0, 0, 0]
        # pass strides

        if (
            layer_param.pooling_param.stride_h != 0
            and layer_param.pooling_param.stride_w != 0
        ):
            strides = [
                layer_param.pooling_param.stride_h,
                layer_param.pooling_param.stride_w,
            ]

        # enum RoundMode {
        #     CEIL = 0;
        #     FLOOR = 1;
        # }

        ceil_mode = True
        if layer_param.pooling_param.round_mode == 0:
            ceil_mode = True
        elif layer_param.pooling_param.round_mode == 1:
            ceil_mode = False
        else:
            logger.error("unsupport RoundMode!")
            exit(-1)

        """
        enum PoolMethod {
            MAX = 0;
            AVE = 1;
            STOCHASTIC = 2;
        }  
        """

        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom] 
        
        if pool_value == 0 and global_value is True:
            node = graph.AdaptiveMaxPool2d(layer_param.name, input_names, output_names)
        elif pool_value == 1 and global_value is True:
            node = graph.AdaptiveAvgPool2d(layer_param.name, input_names, output_names)
        elif pool_value == 0 and global_value is False:
            node = graph.MaxPool2d(layer_param.name, input_names, output_names)
            node.kernel_size = kernel_size
            node.stride = strides
            node.padding = pads
            node.ceil_mode = ceil_mode
        elif pool_value == 1 and global_value is False:
            node = graph.AvgPool2d
            node.kernel_size = kernel_size
            node.stride = strides
            node.padding = pads
            node.ceil_mode = ceil_mode
        else:
            logger.error("unsupport pooling!")
            exit(-1)
        return node
