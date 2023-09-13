import torch
import torch.utils.data as data
import numpy as np
import os


class DFaustDataSet(data.Dataset):

    def __init__(self, dataset_path, split_path, points_batch=10000, d_in=3, with_normals=False, grid_range=1.1,
                 val=False, part_i=None):
        # dataset_path, gt_path, scan_path
        self.grid_range = grid_range

        print('getting npy files')
        dataset_path = os.path.expanduser(dataset_path)
        self.dataset_path = dataset_path
        assert os.path.exists(dataset_path)
        split_path = os.path.expanduser(split_path)
        assert os.path.exists(split_path), split_path
        with open(split_path, "r") as f:
            split = f.readlines()
        # self.npyfiles_mnfld = get_instance_filenames(dataset_path, split)  # list of filenames
        self.npyfiles_mnfld = open(os.path.join(dataset_path, split_path), 'r+').readlines()
        print(len(self.npyfiles_mnfld), 'npy files found')
        # splitting the dataset into 15 parts for parallel training
        # idx = int(np.ceil(len(self.npyfiles_mnfld) / 15))
        # if part_i is not None:
        #     self.npyfiles_mnfld = self.npyfiles_mnfld[part_i * idx:(part_i + 1) * idx]
        self.n_points = points_batch
        self.with_normals = with_normals
        self.d_in = d_in

    def load_points(self, index):
        return np.load(os.path.join(self.dataset_path, self.npyfiles_mnfld[index].strip())) # (250000, 6) which has xyz, normal xyz
        # return np.load(self.npyfiles_mnfld[index])  # (N, 6) which has xyz, normal xyz

    def get_info(self, index):
        shape_name, pose, tag = self.npyfiles_mnfld[index].split('/')[-3:]
        return shape_name, pose, tag[:tag.find('.npy')]

    def __getitem__(self, index):
        # index = torch.tensor([0]).long()
        point_set_mnlfld = torch.from_numpy(self.load_points(index)).float()  # (250000, 6) which has xyz, normal xyz

        random_idx = torch.randperm(point_set_mnlfld.shape[0])[:self.n_points]
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx)  # (pnts, 6)

        mnfld_points = point_set_mnlfld[:, :self.d_in]

        if self.with_normals:
            normals = point_set_mnlfld[:, -self.d_in:]  # todo adjust to case when we get no sigmas
        else:
            normals = torch.empty(0)

        # Q_far
        nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                                            size=(self.n_points, 3)).astype(np.float32)  # (n_points, 3)
        nonmnfld_points = torch.from_numpy(nonmnfld_points).float()

        # Q_near
        dist = torch.cdist(mnfld_points, mnfld_points)
        sigmas = torch.topk(dist, k=51, dim=1, largest=False)[0][:, -1:]  # (n_points, 1)
        near_points = (mnfld_points + sigmas * torch.randn(mnfld_points.shape[0],
                                                           mnfld_points.shape[1]))
        return {'mnfld_points': mnfld_points, 'mnfld_n': normals, 'nonmnfld_points': nonmnfld_points,
                'near_points': near_points, 'indices': index, 'name': self.npyfiles_mnfld[index]}

    def __len__(self):
        return len(self.npyfiles_mnfld)


def get_instance_filenames(base_dir, split):
    npyfiles = []
    for line in split:
        line = line.strip()
        npyfiles.append(os.path.join(base_dir, line))
    return npyfiles
