#!/bin/bash

export nnUNet_n_proc_DA=24
CUDA_VISIBLE_DEVICES=3 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC Task554_LungNoduleSE 0 --disable_next_stage_pred > logs/3d_lowres_dtc_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC Task554_LungNoduleSE 1 --disable_next_stage_pred > logs/3d_lowres_dtc_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC Task554_LungNoduleSE 2 --disable_next_stage_pred -val > logs/3d_lowres_dtc_2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC Task554_LungNoduleSE 3 --disable_next_stage_pred -val > logs/3d_lowres_dtc_3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC Task554_LungNoduleSE 4 --disable_next_stage_pred -val > logs/3d_lowres_dtc_4.log 2>&1 &

