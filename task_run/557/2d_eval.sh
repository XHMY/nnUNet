#!/bin/bash

nohup nnUNet_evaluate_folder \
-ref "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/labelsTr" \
-pred "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTsPred" \
-l 1 \
>logs/2d_eval.log 2>&1 &