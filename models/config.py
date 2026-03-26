# -*- coding: utf-8 -*-

# Modified by Yan Wang based on the following repositories.
# RTFNet: https://github.com/yuxiangsun/RTFNet
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg


from detectron2.config import CfgNode as CN

def add_dit_config(cfg):
    """
    Add config for DiT_backbone.
    """
    
    cfg.MODEL.DIT = CN()
    cfg.MODEL.DIT.INPUT_SIZE=(320, 416)
    cfg.MODEL.SWIN.USE_DIT_BRANCH = True
    cfg.MODEL.DIT.PATCH_SIZE = 32
    cfg.MODEL.DIT.IN_CHANNELS = 4
    cfg.MODEL.DIT.COND_CHANNELS = 4
    cfg.MODEL.DIT.COND_EMB_CHANNELS = 1
    cfg.MODEL.DIT.HIDDEN_SIZE = 128
    cfg.MODEL.DIT.DEPTH = 3
    cfg.MODEL.DIT.NUM_HEADS = 4
    cfg.MODEL.DIT.MLP_RATIO = 4.0
    cfg.MODEL.DIT.TIMESTEPS = 100
    cfg.MODEL.DIT.OBJECTIVE = "pred_noise"
    cfg.MODEL.DIT.SAMPLING_STEPS=1
    cfg.MODEL.DIT.CKPT_PATH = './pretrained_model/SegDiT_ckpt/segdit_best.pth'