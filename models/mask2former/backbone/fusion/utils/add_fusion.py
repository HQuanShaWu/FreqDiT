import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import tqdm
from scipy.io import loadmat, savemat
from nets.fusion_model import SwinFusion
import torch


def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def normalize(data):
    return data / 255.


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


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
        torch.load('../checkpoint_mat_rgb/70_E.pth'))
    print(key)
    netG.eval()
    return netG


def read(img_A, img_B):
    img_A_all = cv2.cvtColor(img_A, cv2.COLOR_RGB2Lab)
    img_A = cv2.cvtColor(img_A, cv2.COLOR_RGB2GRAY)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_RGB2GRAY)
    img_A = normalize(img_A)[..., None]
    img_B = normalize(img_B)[..., None]
    img_A_all = normalize(img_A_all)
    return img_A, img_B, img_A_all


def run(img_a, img_b, image_lab):
    window_size = 8
    scale = 1
    with torch.no_grad():
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        img_lab = image_lab.to(device)
        _, _, h_old, w_old = img_a.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
        img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
        img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
        img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]

        img_lab = torch.cat([img_lab, torch.flip(img_lab, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lab = torch.cat([img_lab, torch.flip(img_lab, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_a, img_b, img_lab)
        output = output[..., :h_old * scale, :w_old * scale]
        output = output.detach()[0].float().cpu()
    output = tensor2uint(output)
    return output


if __name__ == '__main__':
    device = 'cuda'
    model = define_model()
    data_path = '/media/zyj/data_all/zyj/done/rgb融合检测/data_use'
    for file in os.listdir(data_path):
        if file == 'train':
            continue
        for file1 in tqdm.tqdm(os.listdir(data_path + '/' + file)):
            print(data_path + '/' + file + '/' + file1)
            data = loadmat(data_path + '/' + file + '/' + file1)
            rgb_a = data['img1']
            lab_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2Lab)

            out = {'img1': data['img1'], 'img2': data['img2']}
            img_A, img_B, img_lab = read(data['img1'], data['img2'])

            img_A = single2tensor3(img_A)
            img_B = single2tensor3(img_B)
            img_lab = single2tensor3(img_lab)

            output = run(img_A[None], img_B[None], img_lab[None])
            output = np.concatenate([output[..., 0:1], lab_a[..., 1:]], axis=-1)
            output = cv2.cvtColor(output, cv2.COLOR_Lab2RGB)

            out['fusion'] = output
            savemat(data_path + '/' + file + '/' + file1, out)
