class MMParameter:
    def __init__(self, data=None):
        self.data = data


class MMNode:
    def __init__(self, name, input_names=[], output_names=[]) -> None:
        self.name = name
        self.input_names = input_names
        self.output_names = output_names

    def construct_code(self):
        type_name = f"nn.{self.__class__.__name__}"
        return f"{type_name}({self.extra_repr()})"

    def create_forward(self):
        args = ",".join(self.input_names)
        ret = ",".join(self.output_names)
        return f"{ret} = self.{self.name}({args})"

    def parameters(self):
        state_dict = {}
        for name, v in self.__dict__.items():
            if isinstance(v, MMParameter):
                state_dict[f'{self.name}.{name}'] = v.data
        return state_dict
    
    def infer_shape(self):
        pass

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        type_name = f"{self.__class__.__name__}"
        return f"{type_name}({self.extra_repr()})"
