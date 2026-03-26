import glob
import os
import random

import torch
from scipy.io import loadmat
from PIL import Image
from fusion.utils.tools import *
from torch.utils.data import Dataset


def get_data_path_mat():
    data_path = '/media/zyj/data_all/zyj/SAM/data_emb/train'
    files = glob.glob(data_path + '/*.mat')
    return files


class Fusion_dataset(Dataset):
    def __init__(self, files, train=True):
        super(Fusion_dataset, self).__init__()
        self.files = files
        self.train = train
        self.patch_size = 128
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def read(self, index):
        A_path = self.files[index][0]
        B_path = self.files[index][1]
        img_A = np.array(Image.open(A_path))
        img_B = np.array(Image.open(B_path))
        img_A = normalize(img_A)[..., None]
        img_B = normalize(img_B)[..., None]
        return img_A, img_B, A_path, B_path

    def train_set(self, index):
        img_A, img_B, A_path, B_path = self.read(index)

        H, W, _ = img_A.shape
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        patch_A = img_A[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        patch_B = img_B[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        mode = random.randint(0, 7)
        img_A, img_B = augment_img(patch_A, mode=mode).copy(), augment_img(patch_B, mode=mode).copy()
        img_A = torch.from_numpy(np.ascontiguousarray(img_A)).permute(2, 0, 1).float()
        img_B = torch.from_numpy(np.ascontiguousarray(img_B)).permute(2, 0, 1).float()
        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def test_set(self, index):
        img_A, img_B, A_path, B_path = self.read(index)
        img_A = single2tensor3(img_A)
        img_B = single2tensor3(img_B)

        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __getitem__(self, index):
        if self.train:
            return self.train_set(index)
        else:
            return self.test_set(index)


class Fusion_dataset_mat(Dataset):
    def __init__(self, files, train=True):
        super(Fusion_dataset_mat, self).__init__()
        self.files = files
        self.train = train
        self.patch_size = 128
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def read(self, index):
        path = self.files[index]
        data = loadmat(path)
        vis = data['vis']
        inf = data['inf']
        vis_emb = data['vis_emb']
        inf_emb = data['inf_emb']

        vis_lab = cv2.cvtColor(vis, cv2.COLOR_RGB2Lab)
        vis = vis_lab[..., 0:1]
        inf = cv2.cvtColor(inf, cv2.COLOR_RGB2GRAY)[..., None]
        vis = normalize(vis).transpose(2, 0, 1)
        inf = normalize(inf).transpose(2, 0, 1)
        return vis_emb, inf_emb, vis, inf, path, path

    def train_set(self, index):
        vis_emb, inf_emb, vis, inf, *_ = self.read(index)
        vis_emb = torch.from_numpy(vis_emb).type(torch.FloatTensor)
        inf_emb = torch.from_numpy(inf_emb).type(torch.FloatTensor)

        vis = torch.from_numpy(vis).type(torch.FloatTensor)
        inf = torch.from_numpy(inf).type(torch.FloatTensor)
        return {'vis_emb': vis_emb, 'inf_emb': inf_emb, 'vis': vis, 'inf': inf}

    def test_set(self, index):
        vis_emb, inf_emb, vis, inf, path, _ = self.read(index)
        vis_emb = torch.from_numpy(vis_emb).type(torch.FloatTensor)
        inf_emb = torch.from_numpy(inf_emb).type(torch.FloatTensor)

        vis = torch.from_numpy(vis).type(torch.FloatTensor)
        inf = torch.from_numpy(inf).type(torch.FloatTensor)
        return {'vis_emb': vis_emb, 'inf_emb': inf_emb, 'vis': vis, 'inf': inf, 'path': path}

    def __getitem__(self, index):
        if self.train:
            return self.train_set(index)
        else:
            return self.test_set(index)


if __name__ == '__main__':
    x = get_data_path_mat()
    print(x)
    sample = loadmat(x[0])
    vis = sample['vis']
    inf = sample['inf']
    vis_emb = sample['vis_emb']
    inf_emb = sample['inf_emb']

    print(vis.shape, vis_emb.shape)

