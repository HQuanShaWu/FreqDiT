import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import copy
from .regularizers import *
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim import Adam
from collections import OrderedDict
from fusion.nets.fusion_model import SwinFusion
from fusion.nets.model_base import ModelBase
from fusion.utils.losses import fusion_loss_vif


class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__()
        self.opt = opt  # opt
        self.save_dir = opt.save_dir  # save models
        self.device = opt.device
        self.is_train = opt.is_train  # training or not

        height = 320
        width = 416
        window_size = 8
        self.netG = SwinFusion(upscale=1,
                               img_size=(height, width),
                               window_size=window_size,
                               img_range=1.,
                               depths=[2, 2, 6, 2],
                               embed_dim=60,
                               num_heads=[6, 6, 6, 6],
                               mlp_ratio=2,
                               upsampler='',
                               resi_connection='1conv').to(device=self.device)
        if self.opt.E_decay > 0:
            self.netE = copy.deepcopy(self.netG).eval()

    def init_train(self):
        # self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        # self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt.G_scheduler_milestones,
                                                        self.opt.G_scheduler_gamma
                                                        ))

    def define_loss(self):
        self.G_lossfn = fusion_loss_vif().to(self.device)
        self.G_lossfn_weight = self.opt.G_lossfn_weight

    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt.G_optimizer_lr, weight_decay=0)

    def feed_data(self, data, need_GT=False, phase='test'):
        self.A = data['vis_emb'].to(self.device)
        self.B = data['inf_emb'].to(self.device)

        self.A_ori = data['vis'].to(self.device)
        self.B_ori = data['inf'].to(self.device)
        if need_GT:
            self.GT = data['GT'].to(self.device)

    def netG_forward(self, phase='test'):
        self.E = self.netG(self.A, self.B)

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        total_loss, loss_text, loss_int, loss_ssim = self.G_lossfn(self.A_ori, self.B_ori, self.E)
        G_loss = self.G_lossfn_weight * total_loss
        G_loss.backward()

        G_optimizer_clipgrad = self.opt.G_optimizer_clipgrad if \
            self.opt.G_optimizer_clipgrad else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()

        G_regularizer_orthstep = self.opt.G_regularizer_orthstep \
            if self.opt.G_regularizer_orthstep else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt.G_regularizer_clipstep \
            if self.opt.G_regularizer_clipstep else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt.checkpoint_save != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['Text_loss'] = loss_text.item()
        self.log_dict['Int_loss'] = loss_int.item()
        self.log_dict['SSIM_loss'] = loss_ssim.item()

        if self.opt.E_decay > 0:
            self.update_E(self.opt.E_decay)
        return self.log_dict

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt.E_decay > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt.G_optimizer_reuse:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
