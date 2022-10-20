from ..node import MMNode, MMParameter


class Conv2d(MMNode):
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.in_channels = None
        self.out_channels = None
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1
        self.bias = True
        self.padding_mode = "zeros"
        self.weight = None
        self.bias = None

    def infer_shape(self):
        pass

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
            ", padding={padding}"
            ", dilation={dilation}"
            ", groups={groups}"
            ", padding_mode='{padding_mode}'"
        )
        if self.bias is None:
            s += ", bias=False"
        else:
            s += ", bias=True"
        return s.format(**self.__dict__)
