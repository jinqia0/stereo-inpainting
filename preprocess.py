from transformers import pipeline
import os
import numpy as np
from PIL import Image
import open3d as o3d
from torchvision import transforms
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


def preprocess_and_save(input_dir, debug=False):
    """
    批量预处理存放左右帧图片的文件夹。

    Args:
        input_dir (str): 输入文件夹路径，包含左右视角帧的子文件夹。
    """
    # 初始化深度估计模型
    try:
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    except Exception as e:
        print(f"Failed to initialize depth estimation pipeline: {e}")
        return

    for video_dir in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_dir)
        if not os.path.isdir(video_path):
            continue

        for view in ['left', 'right']:
            view_path = os.path.join(video_path, view)
            if not os.path.isdir(view_path):
                continue

            # 创建所需的目录
            dirs = create_dirs(view_path, ['depth', 'wrapped_frames', 'masks'])

            # 遍历帧图像
            frames_dir = os.path.join(view_path, 'frames')
            if not os.path.isdir(frames_dir):
                continue

            for image_name in os.listdir(frames_dir):
                try:
                    # 加载图像
                    image_path = os.path.join(frames_dir, image_name)
                    image = Image.open(image_path).convert("RGB")

                    # 深度估计
                    depth_image = pipe(image)['depth']

                    # 计算相对视角（如果当前是 left，则传入 right；如果当前是 right，则传入 left）
                    opposite_view = 'right' if view == 'left' else 'left'

                    # PCD 构建与重新渲染
                    wrapped_image_array = get_rgb_from_view(
                        image, depth_image, view=opposite_view, device=o3d.core.Device("CPU:0")
                    )
                    wrapped_image = Image.fromarray(wrapped_image_array.astype(np.uint8))

                    # 创建掩码
                    mask_array = np.all(wrapped_image_array == 0, axis=-1)
                    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))

                    # 保存结果
                    save_image(depth_image, os.path.join(dirs['depth'], image_name))
                    save_image(wrapped_image, os.path.join(dirs['wrapped_frames'], image_name))
                    save_image(mask_image, os.path.join(dirs['masks'], image_name))

                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
                    continue

                if debug:
                    return


def main():
    input_dir = 'dataset/Monkaa'

    preprocess_and_save(input_dir, debug=False)


if __name__ == '__main__':
    main()
