from dataset.realworld import *
path = 'data/RISE_rohand'
dataset = RealWorldDataset(path=path, cam_ids=['038522063145'], split="train")
# import pdb; pdb.set_trace()
# dataset.__getitem__(0)
# for i in range(len(dataset)):
#     print(i)
#     dataset.__getitem__(i)
# import pdb;pdb.set_trace()