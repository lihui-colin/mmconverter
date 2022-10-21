from ..node import MMNode, MMParameter


class Reshape(MMNode):
    shortname = "reshape"
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)
        self.shape = []

    def construct_code(self):
        return None

    def extra_repr(self):
        s = "{shape}"
        return s.format(**self.__dict__)

    def create_forward(self):
        args = "*".join(self.input_names)
        ret = ",".join(self.output_names)
        str_shape = ",".join([str(x) for x in self.shape])
        return f"{ret} = {args}.reshape({str_shape})"
