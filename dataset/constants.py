import numpy as np

from utils.constants import *



TO_TENSOR_KEYS = [
    "input_coords_list",
    "input_feats_list",
    "action",
    "action_normalized",
]

# camera intrinsics
INTRINSICS = {
    "043322070878": np.array(
        [
            [909.72656250, 0, 645.75042725, 0],
            [0, 909.66497803, 349.66162109, 0],
            [0, 0, 1, 0],
        ]
    ),
    "750612070851": np.array(
        [
            [922.37457275, 0, 637.55419922, 0],
            [0, 922.46069336, 368.37557983, 0],
            [0, 0, 1, 0],
        ]
    ),
    "104422070044": np.array(
        [
            [910.40948486, 0.0, 648.34844971, 0.0],
            [0.0, 908.31109619, 371.22473145, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    "038522063145": np.array(
        [
            [927.76538086, 0.0, 649.1932373, 0.0],
            [0.0, 927.45159912, 367.11340332, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
}

# inhand camera serial
INHAND_CAM = ["104422070044"]

# transformation matrix from inhand camera (corresponds to INHAND_CAM[0]) to tcp
INHAND_CAM_TCP = np.array(
    [[0, -1, 0, 0], [1, 0, 0, 0.077], [0, 0, 1, 0.1865], [0, 0, 0, 1]]
)
