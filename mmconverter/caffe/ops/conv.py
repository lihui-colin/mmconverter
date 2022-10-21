from typing import OrderedDict
from ...builder import CAFFEOPS as OPS
from ... import graph
import numpy as np
import torch


@OPS.register_module()
class Convolution:
    shortname = "conv"

    def __init__(self) -> None:
        pass

    def __call__(self, layer_param, params) -> graph.Conv2d:
        # 膨胀系数 dilations
        dilation = [1]
        if layer_param.convolution_param.dilation != []:
            dilation = layer_param.convolution_param.dilation
        if len(dilation) == 1:
            strides = dilation[0]

        ##填充pads
        pads = [0]  # 默认为0
        if layer_param.convolution_param.pad != []:  # 若存在pad,则根据pad赋值
            pads = layer_param.convolution_param.pad 
        elif (
            layer_param.convolution_param.pad_h != 0
            or layer_param.convolution_param.pad_w != 0
        ):  # 若存在pad_w,pad_h则根据其赋值
            pads = [
                layer_param.convolution_param.pad_h,
                layer_param.convolution_param.pad_w,
            ]
        if len(pads) == 1:
                pads = pads[0]

        ##步长strides
        strides = [1]  # 默认为1
        if layer_param.convolution_param.stride != []:
            strides = layer_param.convolution_param.stride 
        elif (
            layer_param.convolution_param.stride_h != 0
            and layer_param.convolution_param.stride_w != 0
        ):
            strides = [
                layer_param.convolution_param.stride_h,
                layer_param.convolution_param.stride_w,
            ]
        if len(strides) == 1:
            strides = strides[0]
            
        ##kernel_size
        if layer_param.convolution_param.kernel_size == []:
            kernel_size = [
                layer_param.convolution_param.kernel_h,
                layer_param.convolution_param.kernel_w,
            ]
        else:
            kernel_size = layer_param.convolution_param.kernel_size
        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        groups = layer_param.convolution_param.group
        bias = layer_param.convolution_param.bias_term

        weight = params[0].data
        output_channels, input_channels = weight.shape[:2]

        output_names = [x for x in layer_param.top]
        input_names = [x for x in layer_param.bottom]
        node = graph.Conv2d(layer_param.name, input_names, output_names)
        node.in_channels = input_channels
        node.out_channels = output_channels
        node.kernel_size = kernel_size
        node.stride = strides
        node.padding = pads
        node.dilation = dilation
        node.groups = groups
        node.padding_mode = "zeros"

        node.weight = graph.MMParameter(weight)

        if bias:
            bias_t = params[1].data
            node.bias = graph.MMParameter(bias_t)
        return node
