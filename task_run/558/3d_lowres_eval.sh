#!/bin/bash

nohup nnUNet_evaluate_folder \
-ref "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/labelsTr" \
-pred "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554_mst" \
-l 1 \
>logs/3d_lowres_554_mst_test_557_eval_3.log 2>&1 &