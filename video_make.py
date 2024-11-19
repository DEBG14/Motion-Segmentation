import cv2
import os
from dotenv import load_dotenv
import numpy as np

# Load paths from .env file
load_dotenv()
foreground_path = os.getenv("OUTPUT_FOREGROUND_PATH")
background_path = os.getenv("OUTPUT_BACKGROUND_PATH")
output_video_path = os.getenv("OUTPUT_VIDEO_PATH")
output_foreground_video = os.path.join(output_video_path, "foreground.mp4")
output_background_video = os.path.join(output_video_path, "background.mp4")


def extract_integer(filename):
    """Extract integer index from the filename for sorting."""
    return int(filename.split('.')[0][1:])


def load_images(image_path):
    """Load images from the directory and return sorted file paths."""
    images = sorted(os.listdir(image_path), key=extract_integer)
    return [os.path.join(image_path, img) for img in images]


def images_to_video(image_files, output_path, fps=30):
    """Convert a sequence of images into a video."""
    if not image_files:
        print(f"No images found in {os.path.dirname(image_files[0])}.")
        return

    # Get video properties from the first image
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape
    size = (width, height)

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for filename in image_files:
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    print(f"Video saved at {output_path}")


# Load and process images
foreground_images = load_images(foreground_path)
background_images = load_images(background_path)

# Create videos from images
os.makedirs(output_video_path, exist_ok=True)  # Ensure output directory exists
images_to_video(foreground_images, output_foreground_video)
images_to_video(background_images, output_background_video)
