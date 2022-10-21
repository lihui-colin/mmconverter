import torch
import numpy as np

class Blob:
    def __init__(self, blob_proto):
        shape = [dim for dim in blob_proto.shape.dim]
        self.data = torch.from_numpy(np.array(blob_proto.data, dtype=np.float32).reshape(shape))
