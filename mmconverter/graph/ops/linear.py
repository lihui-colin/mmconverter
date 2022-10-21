from ..node import MMNode, MMParameter


class Linear(MMNode):
    shortname = "linear"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.in_features = None
        self.out_features = None
        self.weight = None
        self.bias = None

    def extra_repr(self):
        s = (
            "{in_features}"
            ", {out_features}"
        )
        if self.bias is not None:
            s += ", bias=True"
        return s.format(**self.__dict__)
