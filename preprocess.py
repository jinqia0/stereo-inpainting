import os
import argparse
from tqdm import tqdm
from transformers import pipeline
from PIL import Image
import numpy as np
import open3d as o3d
from utils_pcd import get_rgb_from_view


def create_dirs(base_dir, sub_dirs):
    """创建多个子目录"""
    paths = {name: os.path.join(base_dir, name) for name in sub_dirs}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_image(image, path, mode="RGB"):
    """保存 PIL 图像"""
    try:
        image.save(path)
    except Exception as e:
        print(f"Failed to save image at {path}: {e}")


def initialize_pipeline():
    """初始化深度估计模型."""
    try:
        return pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    except Exception as e:
        print(f"Failed to initialize depth estimation pipeline: {e}")
        return None


def create_processing_dirs(view_path):
    """创建所需的子目录."""
    return create_dirs(view_path, ['depth', 'wrapped_frames', 'masks'])


def process_frame(image_path, dirs, pipe, view, device="CPU:0"):
    """
    处理单帧图像：深度估计、视角映射、掩码生成，并保存结果.

    Args:
        image_path (str): 图像路径.
        dirs (dict): 保存路径字典.
        pipe (Pipeline): 深度估计模型.
        view (str): 当前视角 ('left' 或 'right').
        device (str): 处理设备.
    """
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 深度估计
        depth_image = pipe(image)['depth']

        # 计算相对视角
        opposite_view = 'right' if view == 'left' else 'left'

        # PCD 构建与重新渲染
        wrapped_image_array = get_rgb_from_view(
            image, depth_image, view=opposite_view, device=o3d.core.Device(device)
        )
        wrapped_image = Image.fromarray(wrapped_image_array.astype(np.uint8))

        # 创建掩码
        mask_array = np.all(wrapped_image_array == 0, axis=-1)
        mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))

        # 保存结果
        save_image(depth_image, os.path.join(dirs['depth'], os.path.basename(image_path)))
        save_image(wrapped_image, os.path.join(dirs['wrapped_frames'], os.path.basename(image_path)))
        save_image(mask_image, os.path.join(dirs['masks'], os.path.basename(image_path)))

    except Exception as e:
        print(f"Error processing image {os.path.basename(image_path)}: {e}")

def process_view(view_path, pipe, view, debug=False):
    """
    处理指定视角的所有帧图像.

    Args:
        view_path (str): 视角路径 ('left' 或 'right').
        pipe (Pipeline): 深度估计模型.
        view (str): 当前视角.
        debug (bool): 调试模式.
    """
    if not os.path.isdir(view_path):
        return

    # 创建子目录
    dirs = create_processing_dirs(view_path)

    # 获取帧图像列表
    frames_dir = os.path.join(view_path, 'frames')
    if not os.path.isdir(frames_dir):
        return

    frame_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]
    with tqdm(frame_files, desc=f"Processing Frames ({view})", unit="frame", leave=False) as frame_pbar:
        for image_path in frame_pbar:
            process_frame(image_path, dirs, pipe, view)

            if debug:  # 调试模式处理一张图片后退出
                return


def process_video(video_path, pipe, debug=False):
    """
    处理单个视频文件夹.

    Args:
        video_path (str): 视频文件夹路径.
        pipe (Pipeline): 深度估计模型.
        debug (bool): 调试模式.
    """
    for view in ['left', 'right']:
        view_path = os.path.join(video_path, view)
        process_view(view_path, pipe, view, debug)


def preprocess_and_save(input_dir, debug=False):
    """
    批量预处理存放左右帧图片的文件夹.

    Args:
        input_dir (str): 输入文件夹路径，包含左右视角帧的子文件夹.
        debug (bool): 调试模式.
    """
    # 初始化模型
    pipe = initialize_pipeline()
    if pipe is None:
        return

    # 获取视频目录列表
    video_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    with tqdm(video_dirs, desc="Processing Videos", unit="video") as video_pbar:
        for video_dir in video_pbar:
            video_path = os.path.join(input_dir, video_dir)
            process_video(video_path, pipe, debug)


def main():
    parser = argparse.ArgumentParser(description="Preprocess and save dataset.")
    parser.add_argument("input_dir", type=str, help="Path to the input dataset directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    args = parser.parse_args()
    preprocess_and_save(args.input_dir, args.debug)


if __name__ == '__main__':
    main()
