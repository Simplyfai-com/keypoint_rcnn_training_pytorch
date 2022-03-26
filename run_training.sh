
DATA_PATH="/media/leiverandres/LeiverSSD/box_band_data/datasets/treadmill_pytorch_dataset/"
python3 custom_train.py \
    --data-path $DATA_PATH \
    --epochs 2 \
    --output-dir "./test_26_mar_2022" \

# python3 train.py \
#  --dataset boxes \
#  --data-path $DATA_PATH \
#  --data-augmentation "custom" \
#  --model keypointrcnn_resnet50_fpn \
#  -b 4 \
#  --epochs 50 \
#  --lr 0.001 \
#  --data-augmentation custom \
#  --output-dir ./custom_test