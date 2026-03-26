import time
import os

import tqdm
from prettytable import PrettyTable
from utils.Metric import Evaluator
from nets.fusion_model import SwinFusion
from torch.utils.data import DataLoader
from utils.tools import *
from utils import data_loader


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
        torch.load('./checkpoint_mat_rgb/25_E.pth'))
    print(key)
    netG.eval()
    return netG


if __name__ == '__main__':
    device = 'cuda'
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
        result_txt = open('./result.txt', 'w')
        img_a = test_data['A'].to(device)
        img_b = test_data['B'].to(device)
        start = time.time()
        with torch.no_grad():
            _, _, h_old, w_old = img_a.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]
            output = model(img_a, img_b)
            output = output[..., :h_old * scale, :w_old * scale]
            output = output.detach()[0].float().cpu()
        end = time.time()
        output = tensor2uint(output)
        img_a = img_a.cpu().numpy()[0, 0]
        img_b = img_b.cpu().numpy()[0, 0]

        if output.shape != img_a.shape:
            output = cv2.resize(output, img_a.shape[::-1])

        EN = Evaluator.EN(output)
        SD = Evaluator.SD(output)
        SF = Evaluator.SF(output)
        MI = Evaluator.MI(output, img_a, img_b)
        SCD = Evaluator.SCD(output, img_a, img_b)
        VIF = Evaluator.VIFF(output, img_a, img_b)
        Qabf = Evaluator.Qabf(output, img_a, img_b)
        SSIM = Evaluator.SSIM(output, img_a, img_b)
        table.add_row([str(i), EN, SD, SF, MI, SCD, VIF, Qabf, SSIM])
        ENs.append(EN)
        SDs.append(SD)
        SFs.append(SF)
        MIs.append(MI)
        SCDs.append(SCD)
        VIFs.append(VIF)
        Qabfs.append(Qabf)
        SSIMs.append(SSIM)
        result_txt.write(str(table))
    result_txt = open('./output/result.txt', 'w')
    table.add_row(['mean', np.mean(ENs), np.mean(SDs), np.mean(SFs), np.mean(MIs),
                   np.mean(SCDs), np.mean(VIFs), np.mean(Qabfs), np.mean(SSIMs)])
    result_txt.write(str(table))
