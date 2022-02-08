#!/bin/bash

# conda activate nnunet
CUDA_VISIBLE_DEVICES=6 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 0 > logs/2d_default_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 1 > logs/2d_default_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 2 > logs/2d_default_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 3 > logs/2d_default_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 4 > logs/2d_default_4.log 2>&1 &