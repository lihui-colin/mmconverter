from ..node import MMNode


class ReLU(MMNode):
    shortname = "relu"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.inplace = False

    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class LeakyReLU(MMNode):
    shortname = "leakyrelu"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.negative_slope = 0.01
        self.inplace = False

    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)
