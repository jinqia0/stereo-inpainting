# -*- coding: utf-8 -*-
import sys 
sys.path.append(".") 

import os
import cv2
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from RAFT import RAFT
from utils.flow_util import *


def initialize_RAFT(model_path='weights/raft-things.pth', device='cuda'):
    """Initializes the RAFT model."""
    model = torch.nn.DataParallel(RAFT())
    model.load_state_dict(torch.load(model_path))
    model = model.module
    model.to(device)
    model.eval()
    return model


def process_frames_in_view(view_path, RAFT_model, device, h_new, w_new):
    """
    Process all frames in a given view (left or right).

    Args:
        view_path (str): Path to the view folder (e.g., 'left' or 'right').
        RAFT_model: Initialized RAFT model.
        device: Processing device (e.g., 'cuda').
        h_new, w_new (int): Resized height and width.
    """
    frames_path = os.path.join(view_path, 'frames')
    flow_path = os.path.join(view_path, 'flow')

    os.makedirs(flow_path, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, f)))

    print(f"Processing frames in {view_path}...")

    # Add tqdm progress bar for frame processing
    with tqdm(total=len(frame_files) - 1, desc=f"Computing Flow ({view_path})", unit="pair") as pbar:
        for i in range(len(frame_files) - 1):
            img1_path = os.path.join(frames_path, frame_files[i])
            img2_path = os.path.join(frames_path, frame_files[i + 1])

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            transform = transforms.Compose([transforms.ToTensor()])
            img1 = transform(img1).unsqueeze(0).to(device)[:, [2, 1, 0], :, :]
            img2 = transform(img2).unsqueeze(0).to(device)[:, [2, 1, 0], :, :]

            img1 = F.interpolate(img1, size=(h_new, w_new), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(h_new, w_new), mode='bilinear', align_corners=False)

            with torch.no_grad():
                img1 = img1 * 2 - 1
                img2 = img2 * 2 - 1

                _, flow_f = RAFT_model(img1, img2, iters=20, test_mode=True)
                _, flow_b = RAFT_model(img2, img1, iters=20, test_mode=True)

            flow_f = flow_f[0].permute(1, 2, 0).cpu().numpy()
            flow_b = flow_b[0].permute(1, 2, 0).cpu().numpy()

            flow_f_path = os.path.join(flow_path, f'{frame_files[i][:-4]}_{frame_files[i + 1][:-4]}_f.flo')
            flow_b_path = os.path.join(flow_path, f'{frame_files[i + 1][:-4]}_{frame_files[i][:-4]}_b.flo')

            flowwrite(flow_f, flow_f_path, quantize=False)
            flowwrite(flow_b, flow_b_path, quantize=False)

            # Update progress bar
            pbar.update(1)


def process_video_folder(video_path, RAFT_model, device, h_new, w_new):
    """
    Process a single video folder.

    Args:
        video_path (str): Path to the video folder (e.g., 'video_1').
        RAFT_model: Initialized RAFT model.
        device: Processing device (e.g., 'cuda').
        h_new, w_new (int): Resized height and width.
    """
    for view in ['left', 'right']:
        view_path = os.path.join(video_path, view)
        if os.path.isdir(view_path):
            print(f'Processing {view_path}...')
            process_frames_in_view(view_path, RAFT_model, device, h_new, w_new)


def main():
    parser = argparse.ArgumentParser(description="Process videos to compute optical flow.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to input directory.")
    parser.add_argument('--height', type=int, default=240, help="Height to resize frames.")
    parser.add_argument('--width', type=int, default=432, help="Width to resize frames.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RAFT_model = initialize_RAFT(device=device)

    # Collect video directories
    video_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    # Add tqdm progress bar for video directories
    with tqdm(total=len(video_dirs), desc="Processing Video Folders", unit="folder") as video_pbar:
        for video_dir in video_dirs:
            video_path = os.path.join(args.input_dir, video_dir)

            process_video_folder(video_path, RAFT_model, device, args.height, args.width)

            # Update the progress bar
            video_pbar.update(1)


if __name__ == '__main__':
    main()
