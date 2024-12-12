import os
import shutil

def move_inpainted_frames(base_dir):
    """
    Traverse the directory structure and move 'inpainted_frames_opencv' directories
    from 'warpped_frames' to its parent directory.

    Args:
        base_dir (str): Root directory of the dataset.
    """
    for root, dirs, files in os.walk(base_dir):
        # Process only 'warpped_frames' directories
        if root.endswith("warpped_frames"):
            inpainted_dir = os.path.join(root, "inpainted_frames_opencv")
            
            if os.path.exists(inpainted_dir):
                parent_dir = os.path.dirname(root)  # Get parent directory of 'warpped_frames'
                target_dir = os.path.join(parent_dir, "inpainted_frames_opencv")

                # Avoid overwriting an existing directory
                if os.path.exists(target_dir):
                    print(f"Target directory {target_dir} already exists. Skipping move.")
                else:
                    shutil.move(inpainted_dir, target_dir)
                    print(f"Moved {inpainted_dir} to {target_dir}")

if __name__ == "__main__":
    base_dir = "/home/jinqiao/stereo-inpainting/Datasets/Kitti/test"  # Replace with the path to your dataset directory
    move_inpainted_frames(base_dir)