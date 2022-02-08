#!/bin/bash

# conda activate nnunet
CUDA_VISIBLE_DEVICES=1 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 0 -val > logs/2d_default_fix_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup nnUNet_train 2d nnUNetTrainerV2 Task557_LungNoduleGE 4 -val > logs/2d_default_fix_4.log 2>&1 &