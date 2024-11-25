import os
import argparse

def move_images(input_dir):
    """将 view_path 中的图片移到 frames_dir 中"""
    for video_dir in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_dir)
        if not os.path.isdir(video_path):
            continue

        for view in ['left', 'right']:
            view_path = os.path.join(video_path, view)
            if not os.path.isdir(view_path):
                continue

            frames_dir = os.path.join(view_path, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

            for file_name in os.listdir(view_path):
                file_path = os.path.join(view_path, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    new_path = os.path.join(frames_dir, file_name)
                    os.rename(file_path, new_path)
                    print(f"Moved: {file_path} -> {new_path}")

    print("Image moving completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move images from view_path to frames_dir.")
    parser.add_argument("input_dir", type=str, help="The input directory containing video folders.")
    args = parser.parse_args()

    move_images(args.input_dir)
