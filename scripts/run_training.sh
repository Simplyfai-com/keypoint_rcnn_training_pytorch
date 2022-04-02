#! /bin/bash

DATA_PATH="/home/ubuntu/datasets/full_treadmill_torch_dataset/"
python3 custom_train.py \
    --data-path $DATA_PATH \
    --epochs 100 \
    --output-dir "./28_mar_2022_fixed_lr" \
