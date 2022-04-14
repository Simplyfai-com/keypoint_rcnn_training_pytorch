#! /bin/bash
DATA_PATH="/media/leiverandres/LeiverSSD/box_band_data/datasets/minimal_dataset/"
MODEL_PATH="/home/leiverandres/Repos/personal_projects/keypoint_rcnn_training_pytorch/trained_models/01_apr_2022_lr_scheduler/last_checkpoint.pth"
python3 inference.py \
    --dataset-path $DATA_PATH \
    --model-path $MODEL_PATH
