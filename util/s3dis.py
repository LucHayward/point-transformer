import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None,
                 transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item or ".npy" in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not f'Area_{test_area}' in item]
        else:
            self.data_list = [item for item in data_list if f'Area_{test_area}' in item or "val" in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyzrgbl, N*7 | xyzir, N*4
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        if data.shape[1] == 7:  # Their S3DIS data, xyzrgbl
            coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        else:  # My format, xyzir, this should be generally applicable, actually.
            coord, feat, label = data[:, 0:3], data[:, 3:-1], data[:, -1]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                                          self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop
