from ..node import MMNode, MMParameter


class Softmax(MMNode):
    shortname = "softmax"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.dim = None

    def extra_repr(self):
        s = "dim={dim}"
        return s.format(**self.__dict__)
