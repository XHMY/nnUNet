#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port=4321 --nproc_per_node=4 run/run_training_DDP.py 3d_fullres nnUNetTrainerV2_DDP Task555_LungNoduleSE 0