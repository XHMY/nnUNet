#!/bin/bash

nohup nnUNet_evaluate_folder \
-ref "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/labelsTr" \
-pred "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTsPred/m555_3d_fullres" \
-l 1 \
>logs/3d_fullres_eval_557.log 2>&1 &