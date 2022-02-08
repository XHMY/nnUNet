#!/bin/bash
# CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_lowres nnUNetTrainerV2 Task555_LungNoduleSE 0 --npz

# CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task555_LungNoduleSE 0 --npz > run_train/logs/3d_lowres_default_0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task555_LungNoduleSE 1 --npz > run_train/logs/3d_lowres_default_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task555_LungNoduleSE 2 --npz > run_train/logs/3d_lowres_default_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task555_LungNoduleSE 3 --npz > run_train/logs/3d_lowres_default_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task555_LungNoduleSE 4 --npz > run_train/logs/3d_lowres_default_4.log 2>&1 &

