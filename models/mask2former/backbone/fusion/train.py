import numpy as np
import torch
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from fusion.utils import data_loader, fit_utils
from torch.utils.data import DataLoader


def init():
    class parser:
        epochs = 1000
        G_optimizer_lr = 2e-5
        batch_size = 8
        device = 'cuda'
        is_train = True
        E_decay = .999
        G_regularizer_orthstep = None
        G_regularizer_clipstep = None
        G_optimizer_clipgrad = None
        G_scheduler_milestones = [
            250000,
            400000,
            450000,
            475000,
            500000
        ]
        G_scheduler_gamma = .5
        G_lossfn_weight = 1.
        G_optimizer_reuse = True

        save_dir = './checkpoint_mat_rgb'
        checkpoint_save = 1

    return parser()


if __name__ == '__main__':
    args = init()
    os.makedirs(args.save_dir, exist_ok=True)
    model = fit_utils.ModelPlain(args)
    model.init_train()

    train_dataset = data_loader.Fusion_dataset_mat(
        data_loader.get_data_path_mat())
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    current_step = 0
    for epoch in range(args.epochs):
        dt_size = len(train_loader.dataset)
        pbar = tqdm(total=dt_size // args.batch_size,
                    desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict,
                    mininterval=0.3)
        G_loss = []
        Text_loss = []
        Int_loss = []
        SSIM_loss = []
        for i, train_data in enumerate(train_loader):
            current_step += 1
            model.update_learning_rate(current_step)

            model.feed_data(train_data, need_GT=False)

            logs = model.optimize_parameters(current_step)
            G_loss.append(logs['G_loss'])
            Text_loss.append(logs['Text_loss'])
            Int_loss.append(logs['Int_loss'])
            SSIM_loss.append(logs['SSIM_loss'])
            pbar.set_postfix(**{
                'G_loss': np.mean(G_loss),
                'Text_loss': np.mean(Text_loss),
                'Int_loss': np.mean(Int_loss),
                'SSIM_loss': np.mean(SSIM_loss),
            })
            pbar.update(1)
        pbar.close()
        if epoch % args.checkpoint_save == 0:
            model.save(epoch)
