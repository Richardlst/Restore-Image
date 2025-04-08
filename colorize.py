import os
import sys
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logger = logging.getLogger('colorize')

# Paths to load the model
DIR = r"C:\Users\thinh\OneDrive\Desktop\Restore image"
PROTOTXT = os.path.join(DIR, r"models/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"models/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"models/colorization_release_v2.caffemodel")

# Log model paths
logger.info(f"PROTOTXT path: {PROTOTXT}")
logger.info(f"POINTS path: {POINTS}")
logger.info(f"MODEL path: {MODEL}")

# Load the Model at module level
logger.info("Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
logger.info("Colorization model loaded successfully")

def colorize_image(image, render_factor=35):
    """
    Colorize a grayscale image using a neural network model
    
    Parameters:
    - image: Input image (BGR format)
    - render_factor: Size factor for rendering (higher = more detail but slower)
    
    Returns:
    - Colorized BGR image
    """
    try:
        # Convert to float32 and scale
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Extract L channel and resize
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Colorize the image
        logger.info("Colorizing the image...")
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # Resize ab result to original size
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # Get original L channel
        L = cv2.split(lab)[0]

        # Merge channels
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # Convert back to BGR and post-process
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        # Convert to uint8
        colorized = (colorized * 255).astype("uint8")
        logger.info("Colorization completed successfully")

        return colorized

    except Exception as e:
        logger.error(f"Error in colorize_image: {str(e)}")
        raise
