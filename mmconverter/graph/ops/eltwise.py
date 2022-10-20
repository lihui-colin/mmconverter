from ..node import MMNode

from enum import Enum


class EltwiseOp(Enum):
    PROD = 0
    SUM = 1
    MAX = 2


class Add(MMNode):
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)

    def __repr__(self):
        type_name = f"{self.__class__.__name__}"
        return f"{type_name}({self.extra_repr()})"

    def construct_code(self):
        return None

    def create_forward(self):
        args = "+".join(self.input_names)
        ret = ",".join(self.output_names)
        return f"{ret} = {args}"


class Mul(MMNode):
    def __init__(self, name, input_names, output_names) -> None:
        super().__init__(name, input_names, output_names)

    def construct_code(self):
        return None
    def create_forward(self):
        args = "*".join(self.input_names)
        ret = ",".join(self.output_names)
        return f"{ret} = {args}"