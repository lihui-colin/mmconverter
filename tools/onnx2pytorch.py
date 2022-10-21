import argparse
import _init_path
import mmconverter
import mmconverter.onnx
from pathlib import Path
import torch
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="convert caffe model to onnx")

    parser.add_argument(
        dest="onnx_file",
        action="store",
        help="the path for prototxt file, the file name must end with .prototxt",
    )
 
    args = parser.parse_args()
    return args


def main(args):
    onnx_path = args.onnx_file 
    name = Path(onnx_path).stem
    graph = mmconverter.onnx.Load(onnx_path, name)
    if graph is None:
        return
    logger.info(f"generate code")
    output_code_file = f"{name.lower()}_onnx.py"
    with open(output_code_file, "w") as f:
        f.write(graph.code())

    logger.info(f"generate weight")
    state_dict = graph.state_dict()

    logger.info(f"save weight")
    output_pt_file = f"{name}_onnx.pt"
    torch.save(state_dict, output_pt_file)
    logger.info(f"Save to {output_code_file}")
    logger.info(f"Save to {output_pt_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
