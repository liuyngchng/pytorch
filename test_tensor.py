#!/usr/bin/python3

import torch
import numpy as np
from torch import Tensor


def get_tensor() -> Tensor:
    """
    build a tensor
    :return:
    """
    # Initializing a Tensor Directly from data
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    # return x_data

    # Initializing a Tensor From a NumPy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    return x_np


if __name__ == "__main__":
    tensor = torch.ones(4, 4)
    print(tensor)
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))