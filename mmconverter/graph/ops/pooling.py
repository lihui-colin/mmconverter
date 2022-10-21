from ..node import MMNode


class MaxPool2d(MMNode):
    shortname = "pool"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.kernel_size = None
        self.stride = None
        self.padding = 0
        self.dilation = 1
        self.return_indices = False
        self.ceil_mode = False

    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}, return_indices={return_indices}".format(
                **self.__dict__
            )
        )


class AdaptiveMaxPool2d(MMNode):
    shortname = "pool"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.output_size = None
        self.return_indices = False

    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        return "output_size={}, return_indices={}".format(
            self.output_size, self.return_indices
        )


class AdaptiveAvgPool2d(MMNode):
    shortname = "pool"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.output_size = None

    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        return "output_size={}".format(self.output_size)


class AvgPool2d(MMNode):
    shortname = "pool"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.kernel_size = None
        self.stride = None
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True
        self.divisor_override = None

    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}, ceil_mode={}, count_include_pad={}, divisor_override={}".format(
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
