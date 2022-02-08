#!/bin/bash

CUDA_VISIBLE_DEVICES=3 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554_mst" \
-t 558 -m 3d_lowres \
--num_parts 5 --part_id 4 \
--num_threads_preprocessing 10 \
>logs/3d_lowres_inference_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554_mst" \
-t 558 -m 3d_lowres \
--num_parts 5 --part_id 0 \
--num_threads_preprocessing 10 \
>logs/3d_lowres_inference_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554_mst" \
-t 558 -m 3d_lowres \
--num_parts 5 --part_id 1 \
--num_threads_preprocessing 10 \
>logs/3d_lowres_inference_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554_mst" \
-t 558 -m 3d_lowres \
--num_parts 5 --part_id 2 \
--num_threads_preprocessing 10 \
>logs/3d_lowres_inference_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup nnUNet_predict \
-i "${nnUNet_raw_data_base}/nnUNet_raw_data/Task557_LungNoduleGE/imagesTs"  \
-o "/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE/3d_lowres_554_mst" \
-t 558 -m 3d_lowres \
--num_parts 5 --part_id 3 \
--num_threads_preprocessing 10 \
>logs/3d_lowres_inference_3.log 2>&1 &