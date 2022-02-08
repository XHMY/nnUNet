#!/bin/bash

export nnUNet_n_proc_DA=18
CUDA_VISIBLE_DEVICES=3 nohup nnUNet_train 3d_fullres nnUNetTrainerV2 Task557_LungNoduleGE 0 -val > logs/3d_train_val_fix_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 3d_fullres nnUNetTrainerV2 Task557_LungNoduleGE 1 -val > logs/3d_train_val_fix_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 3d_fullres nnUNetTrainerV2 Task557_LungNoduleGE 2 -val > logs/3d_train_val_fix_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup nnUNet_train 3d_fullres nnUNetTrainerV2 Task557_LungNoduleGE 3 -val > logs/3d_train_val_fix_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup nnUNet_train 3d_fullres nnUNetTrainerV2 Task557_LungNoduleGE 4 -val > logs/3d_train_val_fix_4.log 2>&1 &