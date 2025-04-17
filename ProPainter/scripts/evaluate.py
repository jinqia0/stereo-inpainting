import random
import sys 
sys.path.append(".")
sys.path.append("ProPainter")

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model

def evaluate_video(video_root, output_dir, i3d_model, device):
    video_dirs = [os.path.join(video_root, d) for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))]
    overall_metrics = []
    all_real_activations = []
    all_generated_activations = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=len(video_dirs), desc="Processing videos") as video_pbar:
        for video_path in video_dirs:
            video_name = os.path.basename(video_path)
            left_dir = os.path.join(video_path, 'left')
            right_dir = os.path.join(video_path, 'right')

            # Define frame paths
            comparisons = [
                ('warpped_frames', 'frames'),
                ('inpainted_frames_opencv', 'frames'),
            ]

            video_metrics = []

            for source_type, target_type in comparisons:
                # Randomly choose a view direction (Left to Right or Right to Left)
                if random.choice([True, False]):
                    psnr_list, ssim_list, real_activations, generated_activations = compare_frames(
                        os.path.join(left_dir, source_type),
                        os.path.join(right_dir, target_type),
                        i3d_model,
                        device,
                    )
                else:
                    psnr_list, ssim_list, real_activations, generated_activations = compare_frames(
                        os.path.join(right_dir, source_type),
                        os.path.join(left_dir, target_type),
                        i3d_model,
                        device,
                    )

                # Append activations for global VFID computation
                all_real_activations.extend(real_activations)
                all_generated_activations.extend(generated_activations)

                video_metrics.append({
                    "source": source_type,
                    "target": target_type,
                    "psnr": np.mean(psnr_list),
                    "ssim": np.mean(ssim_list),
                })

            overall_metrics.append({"video_name": video_name, "metrics": video_metrics})
            video_pbar.update(1)

    # Calculate global VFID
    overall_vfid = calculate_vfid(all_real_activations, all_generated_activations)

    # Save overall metrics
    overall_output_path = os.path.join(output_dir, "overall_metrics.json")
    with open(overall_output_path, "w") as f:
        json.dump({"overall_metrics": overall_metrics, "overall_vfid": overall_vfid}, f, indent=4)

    # Print summary
    print("\nSummary of Results:")
    print(f"Overall VFID: {overall_vfid:.3f}")
    for result in overall_metrics:
        print(f"Video: {result['video_name']}")
        for metric in result["metrics"]:
            print(f"  Source: {metric['source']} -> Target: {metric['target']}")
            print(f"    PSNR: {metric['psnr']:.2f}, SSIM: {metric['ssim']:.4f}")

def compare_frames(source_path, target_path, i3d_model, device, size=(432, 240), sample_size=16):
    source_files = sorted(os.listdir(source_path))
    target_files = sorted(os.listdir(target_path))

    # # Randomly sample frames
    # sampled_indices = sorted(random.sample(range(len(source_files)), min(sample_size, len(source_files))))
    # source_files = [source_files[i] for i in sampled_indices]
    # target_files = [target_files[i] for i in sampled_indices]

    psnr_list = []
    ssim_list = []

    source_images = []
    target_images = []

    with tqdm(total=len(source_files), desc="Processing frames", leave=False) as frame_pbar:
        for src_file, tgt_file in zip(source_files, target_files):
            src_img = np.array(Image.open(os.path.join(source_path, src_file)).resize(size, Image.LANCZOS))
            tgt_img = np.array(Image.open(os.path.join(target_path, tgt_file)).resize(size, Image.LANCZOS))

            # Compute PSNR and SSIM
            psnr, ssim = calc_psnr_and_ssim(src_img, tgt_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # Store for VFID computation
            source_images.append(Image.fromarray(src_img.astype(np.uint8)))
            target_images.append(Image.fromarray(tgt_img.astype(np.uint8)))

            frame_pbar.update(1)

    # Compute I3D activations
    real_activations, generated_activations = calculate_i3d_activations(target_images, source_images, i3d_model, device)

    return psnr_list, ssim_list, real_activations, generated_activations


# Example usage
video_root = "/home/jinqiao/stereo-inpainting/Datasets/Kitti/test"
output_dir = "/home/jinqiao/stereo-inpainting/ProPainter/evaluation_results"
device = "cuda"  # or "cpu"
i3d_model = init_i3d_model('/home/jinqiao/stereo-inpainting/ProPainter/weights/i3d_rgb_imagenet.pt')

evaluate_video(video_root, output_dir, i3d_model, device)
