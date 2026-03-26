# CUDA_VISIBLE_DEVICES=0,1 python train.py --config-file "./configs/FMBdataset/swin_v2/swin_v2_tiny.yaml" --num-gpus 2 --name dit_peafusion_fmb_gpu_01

CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-file "./configs/MFdataset/swin_v2/swin_v2_large.yaml" \
    --num-gpus 1 \
    --name dit_largeswin_mfnet_gpu_2 \
    #  --resume_ckpt_path

# CUDA_VISIBLE_DEVICES=2,3 python train.py \
#     --config-file "./configs/PSTdataset/swin_v2/swin_v2_tiny.yaml" \
#     --num-gpus 2 \
#     --name dit_tinyswin_pst900_gpu_23  \
    # --resume_ckpt_path checkpoints/dit_tinyswin_pst900_gpu_23/checkpoint_epoch=19_step=1480.ckpt