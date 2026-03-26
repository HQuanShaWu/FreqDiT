import time
import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tqdm
from prettytable import PrettyTable
from fusion.nets.fusion_model import SwinFusion
from torch.utils.data import DataLoader
from fusion.utils.tools import *
from fusion.utils import data_loader


def define_model():
    height = 128
    width = 128
    window_size = 8
    netG = SwinFusion(upscale=1,
                      img_size=(height, width),
                      window_size=window_size,
                      img_range=1.,
                      depths=[6, 6, 6, 6],
                      embed_dim=60,
                      num_heads=[6, 6, 6, 6],
                      mlp_ratio=2,
                      upsampler='',
                      resi_connection='1conv').to(device)
    key = netG.load_state_dict(
        torch.load('./checkpoint_mat_rgb/700_E.pth'))
    print(key)
    netG.eval()
    return netG


if __name__ == '__main__':
    device = 'cuda:1'
    model = define_model()
    # path = ['']
    test_dataset = data_loader.Fusion_dataset_mat(
        data_loader.get_data_path_mat(), train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
    )
    window_size = 8
    scale = 1
    table = PrettyTable(['id', 'EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM'])
    MIs, ENs, VIFs, SDs, SCDs, Qabfs, SSIMs, SFs = [], [], [], [], [], [], [], []
    for i, test_data in enumerate(tqdm.tqdm(test_loader)):
        A = test_data['vis_emb'].to(device)
        B = test_data['inf_emb'].to(device)

        A_ori = loadmat(test_data['path'][0])['vis']
        A_lab = cv2.cvtColor(A_ori, cv2.COLOR_RGB2Lab)
        B_ori = test_data['inf']
        start = time.time()
        with torch.no_grad():

            output = model(A, B)
            output = output.detach()[0].float().cpu()
        end = time.time()
        output = tensor2uint(output)[..., None]
        output = np.concatenate([output, A_lab[..., 1:]], axis=-1)
        output = cv2.cvtColor(output, cv2.COLOR_Lab2RGB)

        plt.figure(figsize=(12, 12))
        plt.subplot(221)
        plt.imshow(A_ori)

        plt.subplot(222)
        plt.imshow(B_ori.cpu().numpy()[0, 0], 'gray')

        plt.subplot(223)
        plt.imshow(output)
        plt.show()
        break


