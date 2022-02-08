#!/bin/bash

export nnUNet_n_proc_DA=18
CUDA_VISIBLE_DEVICES=3 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task554_LungNoduleSE 0 > logs/3d_lowres_default_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task554_LungNoduleSE 1 > logs/3d_lowres_default_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task554_LungNoduleSE 2 > logs/3d_lowres_default_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task554_LungNoduleSE 3 > logs/3d_lowres_default_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup nnUNet_train 3d_lowres nnUNetTrainerV2 Task554_LungNoduleSE 4 > logs/3d_lowres_default_4.log 2>&1 &

