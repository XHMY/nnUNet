#!/bin/bash

CUDA_VISIBLE_DEVICES=4 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTsPred/m555_3d_fullres" \
-t 555 -m 3d_fullres \
--num_parts 4 --part_id 0 \
--num_threads_preprocessing 10 \
>logs/3d_fullres_inference_557_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTsPred/m555_3d_fullres" \
-t 555 -m 3d_fullres \
--num_parts 4 --part_id 1 \
--num_threads_preprocessing 10 \
>logs/3d_fullres_inference_557_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTsPred/m555_3d_fullres" \
-t 555 -m 3d_fullres \
--num_parts 4 --part_id 2 \
--num_threads_preprocessing 10 \
>logs/3d_fullres_inference_557_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTsPred/m555_3d_fullres" \
-t 555 -m 3d_fullres \
--num_parts 4 --part_id 3 \
--num_threads_preprocessing 10 \
>logs/3d_fullres_inference_557_3.log 2>&1 &

