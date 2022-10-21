from os import access
from . import ops


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class MMGraph:
    def __init__(self, name) -> None:
        self.name = name
        self._nodes = []
        self._dirty = False
        self.input_nodes = []
        self.output_nodes = []

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, i):
        return self._nodes[i]

    def addNode(self, node):
        self._nodes.append(node)
        self._dirty = True

    def sanitizeGraph(self):
        self.input_nodes = []
        for node in self._nodes:
            if isinstance(node, ops.Input):
                self.input_nodes.append(node)

        node_output_names = []
        node_input_names = []
        for node in self._nodes:
            node_output_names += node.output_names
            node_input_names += node.input_names

        node_output_names = set(node_output_names)
        node_input_names = set(node_input_names)
        graph_output_names = list(
            node_output_names - (node_input_names & node_output_names)
        )

        output_nodes = []
        for node in self._nodes:
            for o_name in node.output_names:
                if o_name in graph_output_names:
                    output_nodes.append(node)
                    break
        self.output_nodes = list(set(output_nodes))

        # TODO
        # 从输出节点开始遍历所有节点，多节点重新排序
        
        for idx, node in enumerate(self._nodes):
            node.index = idx
        visited_flag = [False] * len(self._nodes)
        new_nodes = []
        for start_node in output_nodes:
            visited_flag[start_node.index] = True

        self._dirty = False

    def code(self):
        if self._dirty:
            self.sanitizeGraph()

        class_name = self.name.capitalize()
        code_lines = [
            "import torch",
            "import torch.nn as nn",
            f"class {class_name}(nn.Module):",
            f"    def __init__(self) -> None:",
            f"        super().__init__()",
        ]
        code = "\n".join(code_lines)
        code += "\n"
        for node in self._nodes:
            s_code = node.construct_code()
            if s_code:
                code += (
                    (8 * " ") + f"self.{node.name}" + "=" + node.construct_code() + "\n"
                )

        input_args = []
        for node in self.input_nodes:
            input_args += node.output_names
        input_args = ",".join(input_args)

        code += (4 * " ") + f"def forward(self, {input_args}):" + "\n"
        for node in self._nodes:
            f_code = node.create_forward()
            if f_code:
                code += (8 * " ") + f_code + "\n"

        ret_vars = []
        for node in self.output_nodes:
            ret_vars += node.output_names
        ret_vars = ",".join(ret_vars)
        code += (8 * " ") + f"return {ret_vars}" + "\n"
        return code

    def state_dict(self):
        state_dict = {}
        for node in self._nodes:
            state_dict.update(node.parameters())
        return state_dict

    def __repr__(self):
        extra_lines = []
        for node in self._nodes:
            node_str = repr(node)
            extra_lines.append(f"{node.name}: {node_str}")
        return "\n".join(extra_lines)
