from ..node import MMNode


class Input(MMNode):
    shortname = "input"
    def __repr__(self):
        return "Input()"

    def construct_code(self):
        return None

    def create_forward(self):
        return None
