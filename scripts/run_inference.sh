#! /bin/bash
DATA_PATH="/media/leiverandres/LeiverSSD/box_band_data/datasets/full_treadmill_pytorch_dataset/"
MODEL_PATH="./trained_models/26_mar_2022/last_checkpoint.pth"
python3 inference.py \
    --dataset-path $DATA_PATH \
    --model-path $MODEL_PATH
