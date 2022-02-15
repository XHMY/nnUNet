#!/bin/bash

export nnUNet_n_proc_DA=14

# CUDA_VISIBLE_DEVICES=3 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC_Unsupervised Task558_LungNoduleGESemi 0 \
# -pretrained_weights "${RESULTS_FOLDER}/nnUNet/3d_lowres/Task554_LungNoduleSE/nnUNetTrainerV2DTC__nnUNetPlansv2.1/fold_0/model_best.model" \
# --disable_next_stage_pred \
# -p "nnUNetPlans_FabiansResUNet_v2.1" \
# > logs/3d_lowres_dtc_semi_558_0.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC_Unsupervised Task558_LungNoduleGESemi 1 \
# -pretrained_weights "${RESULTS_FOLDER}/nnUNet/3d_lowres/Task554_LungNoduleSE/nnUNetTrainerV2DTC__nnUNetPlansv2.1/fold_1/model_best.model" \
# --disable_next_stage_pred \
# -p "nnUNetPlans_FabiansResUNet_v2.1" \
# > logs/3d_lowres_dtc_semi_558_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC_Unsupervised Task558_LungNoduleGESemi 2 \
-pretrained_weights "${RESULTS_FOLDER}/nnUNet/3d_lowres/Task554_LungNoduleSE/nnUNetTrainerV2DTC__nnUNetPlansv2.1/fold_2/model_best.model" \
--disable_next_stage_pred \
-p "nnUNetPlans_FabiansResUNet_v2.1" \
> logs/3d_lowres_dtc_semi_558_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC_Unsupervised Task558_LungNoduleGESemi 3 \
-pretrained_weights "${RESULTS_FOLDER}/nnUNet/3d_lowres/Task554_LungNoduleSE/nnUNetTrainerV2DTC__nnUNetPlansv2.1/fold_3/model_best.model" \
--disable_next_stage_pred \
-p "nnUNetPlans_FabiansResUNet_v2.1" \
> logs/3d_lowres_dtc_semi_558_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup nnUNet_train 3d_lowres nnUNetTrainerV2DTC_Unsupervised Task558_LungNoduleGESemi 4 \
-pretrained_weights "${RESULTS_FOLDER}/nnUNet/3d_lowres/Task554_LungNoduleSE/nnUNetTrainerV2DTC__nnUNetPlansv2.1/fold_4/model_best.model" \
--disable_next_stage_pred \
-p "nnUNetPlans_FabiansResUNet_v2.1" \
> logs/3d_lowres_dtc_semi_558_4.log 2>&1 &

