from dataset.realworld import *
import torch
path = 'data/RISE_rohand'
dataset = RealWorldDataset(path=path, cam_ids=['038522063145'], split="train")
# import pdb; pdb.set_trace()
# dataset.__getitem__(0)

for i in range(len(dataset)):
# for i in range(50):
    print(i)
    dataset.__getitem__(i)
mins=np.stack(dataset.min_buffer).min(axis=0)
maxs=np.stack(dataset.max_buffer).max(axis=0)
print("Mins ",mins)
print("Maxs ",maxs)
import pdb;pdb.set_trace()