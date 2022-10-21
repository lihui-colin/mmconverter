import argparse
import _init_path
import mmconverter
import mmconverter.caffe
from pathlib import Path
import torch
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="convert caffe model to onnx")

    parser.add_argument(
        dest="proto_file",
        action="store",
        help="the path for prototxt file, the file name must end with .prototxt",
    )

    parser.add_argument(
        dest="caffe_model_file",
        action="store",
        help="the path for caffe model file, the file name must end with .caffemodel!",
    )
    args = parser.parse_args()
    return args


def main(args):
    caffe_graph_path = args.proto_file
    caffe_params_path = args.caffe_model_file
    name = Path(caffe_graph_path).stem
    graph = mmconverter.caffe.Load(caffe_graph_path, caffe_params_path, name)
    if graph is None:
        return
    logger.info(f"generate code")
    output_code_file = f"{name.lower()}.py"
    with open(output_code_file, "w") as f:
        f.write(graph.code())

    logger.info(f"generate weight")
    state_dict = graph.state_dict()

    logger.info(f"save weight")
    output_pt_file = f"{name}.pt"
    torch.save(state_dict, output_pt_file)
    logger.info(f"Save to {output_code_file}")
    logger.info(f"Save to {output_pt_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
