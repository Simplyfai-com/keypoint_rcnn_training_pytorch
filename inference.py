import argparse
import pathlib

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from boxes_dataset import BoxesDataset
from custom_train import get_model, visualize
from utils import collate_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="", help="Path to dataset")
    parser.add_argument("--model-path", type=str, default="", help="Path to model")
    parser.add_argument("--output-folder", type=str, default="inference_output", help="Name of the output folder")
    return parser.parse_args()

def main(args):
    DATASET_PATH = pathlib.Path(args.dataset_path)
    KEYPOINTS_FOLDER_TRAIN = DATASET_PATH / "train"
    KEYPOINTS_FOLDER_TEST = DATASET_PATH / "test"
    INFERENCE_OUTPUT_PATH = DATASET_PATH / args.output_folder
    INFERENCE_OUTPUT_PATH.mkdir(exist_ok=True)

    print("Loading dataset... ", end="")
    dataset_train = BoxesDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
    dataset_test = BoxesDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print("Done.")

    print("Loading model... ", end="")
    model = get_model(num_keypoints=8)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print("Done.")

    THRESHOLD = 0.5
    with torch.no_grad():
        model.cuda()
        model.eval()
        for i, (images, targets) in enumerate(data_loader_train):
            images = [image.cuda() for image in images]
            inference = model(images)
            # Change channel order from (N, C, H, W) to (N, H, W, C) and denormalize
            image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = inference[0]["scores"].detach().cpu().numpy()
            # Indexes of boxes with scores > threshold
            high_scores_idx = np.where(scores > THRESHOLD)[0].tolist()
            # Indexes of boxes left after applying NMS (iou_threshold=0.3)
            post_nms_idxs = torchvision.ops.nms(inference[0]["boxes"][high_scores_idx], inference[0]["scores"][high_scores_idx], iou_threshold=0.3).cpu().numpy()

            keypoints = []
            for kpts in inference[0]["keypoints"][high_scores_idx][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kpts])
                for kp in kpts:
                    if kp[2] == 0:
                        print("Visibility flag: ", kp[2])
            bboxes = []
            for bbox in inference[0]["boxes"][high_scores_idx][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))
            visualize(image, bboxes, keypoints, output_name=f"{INFERENCE_OUTPUT_PATH}/train_{i}.png")
    print("Done.")
                
if __name__ == "__main__":
    args = get_args()
    main(args)
