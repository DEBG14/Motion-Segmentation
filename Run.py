from BG_SUB import SG_model
import cv2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths from .env
foreground_path = os.getenv("OUTPUT_FOREGROUND_PATH")
background_path = os.getenv("OUTPUT_BACKGROUND_PATH")

# Ensure directories exist
os.makedirs(foreground_path, exist_ok=True)
os.makedirs(background_path, exist_ok=True)

# Initialize GMM and video capture
GMM = SG_model(0.008, 0.5, 3)
print("Start")
GMM.parameter_init()
print("Initialization complete")

# Video processing
succ = 1
count = 0
cap = cv2.VideoCapture('umcp.mpg')

while succ:
    succ, frame = cap.read()
    if not succ:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fore, back = GMM.fit(grayscale, frame)

    # Save images to respective paths
    cv2.imwrite(os.path.join(foreground_path, f"f{count}.jpg"), fore)
    cv2.imwrite(os.path.join(background_path, f"b{count}.jpg"), back)
    count += 1
