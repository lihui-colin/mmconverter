from ..node import MMNode, MMParameter


class BatchNorm2d(MMNode):
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.num_features = None
        self.eps = 1e-05
        self.momentum = 0.1
        self.affine = True
        self.track_running_stats = True
        self.weight = None
        self.bias = None
        self.running_mean = None
        self.running_var = None

    def extra_repr(self):
        s = (
            "{num_features}"
            ", eps={eps}"
            ", momentum={momentum}"
            ", affine={affine}"
            ", track_running_stats={track_running_stats}" 
        )
        return s.format(**self.__dict__)


class Scale(MMNode):
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)

        self.weight = None
        self.bias = None
