#!/bin/bash

nohup nnUNet_evaluate_folder \
-ref "${nnUNet_raw_data_base}/nnUNet_raw_data/Task554_LungNoduleSE/labelsTr" \
-pred "/home/lvwei/project/MedicalSegmentation/data/Pred/Task554_LungNoduleSE/3d_lowres_557" \
-l 1 \
>logs/3d_lowres_eval_554.log 2>&1 &