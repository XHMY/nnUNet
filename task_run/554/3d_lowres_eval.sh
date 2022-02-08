#!/bin/bash

# nohup nnUNet_evaluate_folder \
# -ref "${nnUNet_raw_data_base}/nnUNet_raw_data/Task554_LungNoduleSE/labelsTr" \
# -pred "/home/lvwei/project/MedicalSegmentation/data/Pred/Task554_LungNoduleSE/3d_lowres_554" \
# -l 1 \
# >logs/3d_lowres_554_test_554_eval.log 2>&1 &


nohup nnUNet_evaluate_folder \
-ref "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/labelsTr" \
-pred "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554" \
-l 1 \
>logs/3d_lowres_554_test_557_eval.log 2>&1 &