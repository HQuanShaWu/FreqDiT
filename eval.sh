# CUDA_VISIBLE_DEVICES=5 python test.py --config-file "./configs/FMBdataset/swin_v2/swin_v2_tiny.yaml" --num-gpus 1 --name dit_tiny_fmb_eval --checkpoint "peafusion_ckpt/FMB/checkpoint_epoch=109_step=8360_IOU=69.5.ckpt"


# CUDA_VISIBLE_DEVICES=3 python test.py --config-file "./configs/MFdataset/swin_v2/swin_v2_large.yaml" --num-gpus 1 --name dit_large_mf_eval --checkpoint "checkpoints/dit_largeswin_mfnet_gpu_2/checkpoint_epoch=134_step=26460.ckpt"


CUDA_VISIBLE_DEVICES=6 python test.py --config-file "./configs/PSTdataset/swin_v2/swin_v2_tiny.yaml" --num-gpus 1 --name dit_tiny_pst_eval --checkpoint "peafusion_ckpt/PST900/tinyswin_checkpoint_epoch=219_step=16280_IOU=89.0.ckpt"