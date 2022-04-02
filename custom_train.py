import argparse
import glob
import json
import os
import pathlib

import albumentations as A  # Library for augmentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import engine
import transforms
import utils
from boxes_dataset import BoxesDataset
from engine import evaluate, train_one_epoch
from utils import collate_fn

NUM_KEYPOINTS = 8
keypoints_classes_ids2names = {
    0 : "bk_top_L",
    1 : "bK_top_R",
    2 : "bk_bttm_L",
    3 : "bk_bttm_R",
    4 : "ft_top_L",
    5 : "ft_top_R",
    6 : "ft_bttm_L",
    7 : "ft_bttm_R"
}

def get_args():
    parser = argparse.ArgumentParser(description="Pytorch keypoint detection training")
    parser.add_argument("--data-path", type=str, help="Dataset root path")
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="Total num of epochs to run training")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--output-dir", default=".", type=str, help="Path to save outputs")
    parser.add_argument("--resume-from", default="", type=str, help="Path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, help="Start epoch")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    return parser.parse_args()


def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None, output_name=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
    cv2.imwrite(output_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)
        
    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

def get_model(num_keypoints):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)        
    return model

def main(args):
    writer = SummaryWriter(log_dir=args.output_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DATASET_PATH = pathlib.Path(args.data_path)
    KEYPOINTS_FOLDER_TRAIN = DATASET_PATH / "train"
    KEYPOINTS_FOLDER_TEST = DATASET_PATH / "test"
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    print("Loading data...")
    dataset_train = BoxesDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
    dataset_test = BoxesDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)
    print(f"Train: {len(dataset_train)} \nTest: {len(dataset_test)}")
    print("Creating data loaders...")
    data_loader_train = DataLoader(dataset_train, batch_size=12, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Creating model...")
    model = get_model(num_keypoints = NUM_KEYPOINTS)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {args.start_epoch}")

    if args.test_only:
        evaluate(model, data_loader_test, device)
        return
    print("Training...")
    for epoch in range(args.start_epoch, args.epochs):
        writer.add_scalar("Epoch", epoch, epoch)
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000, tb_writer=writer)
        lr_scheduler.step()
        writer.add_scalar("Learning rate", lr_scheduler.get_last_lr()[0], epoch)
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch
            }
            # only keep 3 checkpoints at a time
            checkpoint_files = sorted(glob.glob(args.output_dir + "/*.pth"))
            if len(checkpoint_files) > 2:
                for old_checkpoint in checkpoint_files[:-2]:
                    os.remove(old_checkpoint)
            if epoch < 10: # just put a trailing zero to epoch number, so the sorting works
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_0{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "last_checkpoint.pth"))
        evaluate(model, data_loader_test, device)
    writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
