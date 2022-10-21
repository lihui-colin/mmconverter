import torch
import numpy as np


class Blob:
    def __init__(self, tensor_proto):
        # FLOAT = 1;   // float
        # UINT8 = 2;   // uint8_t
        # INT8 = 3;    // int8_t
        # UINT16 = 4;  // uint16_t
        # INT16 = 5;   // int16_t
        # INT32 = 6;   // int32_t
        # INT64 = 7;   // int64_t
        # STRING = 8;  // string
        # BOOL = 9;    // bool
        shape = [dim for dim in tensor_proto.dims]
        if tensor_proto.data_type == 1:
            self.data = torch.from_numpy(
                np.array(tensor_proto.float_data, dtype=np.float32).reshape(shape)
            )
        elif tensor_proto.data_type == 7:
            self.data = torch.from_numpy(
                np.array(tensor_proto.int64_data, dtype=np.longlong).reshape(shape)
            )
        else:
            assert False, tensor_proto.data_type