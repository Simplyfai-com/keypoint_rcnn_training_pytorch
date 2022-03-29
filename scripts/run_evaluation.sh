#! /bin/bash
DATA_PATH="/media/leiverandres/LeiverSSD/box_band_data/datasets/full_treadmill_pytorch_dataset/"
MODEL_PATH="./trained_models/26_mar_2022/last_checkpoint.pth"
python3 custom_train.py \
    --data-path $DATA_PATH \
    --resume-from $MODEL_PATH \
    --test-only