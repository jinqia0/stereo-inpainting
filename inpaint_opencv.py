import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def process_image(file_path, output_dir):
    """
    Process a single image: read the image, create mask, apply inpainting (GPU fallback to CPU on failure), and save the result.

    Args:
        file_path (str): Path of the image to process.
        output_dir (str): Output directory where the processed image will be saved.
    """
    try:
        print(f"Processing image: {file_path}")

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read {file_path}")
            return

        # Create grayscale image and generate mask
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray_image == 0, 255, 0).astype(np.uint8)

        try:
            # Attempt to upload to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_mask = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            gpu_mask.upload(mask)

            # Attempt GPU inpainting
            inpainted_gpu = cv2.cuda.inpaint(gpu_image, gpu_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            # Download result
            inpainted_image = inpainted_gpu.download()
            print("Inpainting with GPU succeeded.")
        except Exception as e:
            # Fallback to CPU inpainting if CUDA fails
            print(f"CUDA processing failed: {e}. Falling back to CPU processing.")
            inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("Inpainting with CPU succeeded.")

        # Save the processed image
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(output_path, inpainted_image)
        print(f"Saved inpainted image to: {output_path}")
    except Exception as e:
        print(f"Failed to process image {file_path}: {e}")


def inpaint_frames(base_dir):
    """
    Traverse the directory structure, apply inpainting to frames in 'warpped_frames',
    and save the results using multithreading and CUDA with CPU fallback.
    
    Args:
        base_dir (str): Root directory of the dataset.
    """
    for root, dirs, files in os.walk(base_dir):
        # Process only 'warpped_frames' directories
        if root.endswith("warpped_frames"):
            print(f"Processing directory: {root}")

            # Create an output directory
            output_dir = os.path.join(os.path.dirname(root), "inpainted_frames_opencv")
            os.makedirs(output_dir, exist_ok=True)

            # Set up a ThreadPoolExecutor to process images in parallel
            with ThreadPoolExecutor() as executor:
                for file_name in files:
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        file_path = os.path.join(root, file_name)
                        executor.submit(process_image, file_path, output_dir)


if __name__ == "__main__":
    base_dir = "/home/jinqiao/stereo-inpainting/Datasets/Kitti/test"  # Replace with the path to your dataset directory
    inpaint_frames(base_dir)
