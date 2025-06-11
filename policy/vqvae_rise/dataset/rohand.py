import os
import json
import torch
import numpy as np

class RoHandDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        for i in range(self.num_demos):
            clip_start=len(self.obs_frame_ids)
            demo_path = os.path.join(self.data_path, self.all_demos[i])
            tcp_path=os.path.join(demo_path,"lowdim","processed_tcp.npz")
            tcp_data=np.load(tcp_path)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx], self.goal[idx]
