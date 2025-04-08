import numpy as np
import cv2
from PIL import Image
import logging
import traceback
from models.resolution import enhance_resolution
from models.image_enhancement import apply_enhancement_pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('enhance')

def super_resolution(image):
    try:
        logger.debug(f"Input image shape: {image.shape if isinstance(image, np.ndarray) else 'PIL Image'}")
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # First apply detail enhancement
        enhanced = apply_enhancement_pipeline(image)
        
        # Convert to PIL Image for super resolution
        enhanced_pil = Image.fromarray(enhanced)
            
        # Apply super resolution using our model
        result = enhance_resolution(enhanced_pil)
            
        # Convert back to numpy if input was numpy
        if isinstance(image, np.ndarray):
            result = np.array(result)
        
        logger.debug(f"Output image shape: {result.shape if isinstance(result, np.ndarray) else 'PIL Image'}")
        
        return result
    except Exception as e:
        logger.error(f"Error in super_resolution: {str(e)}")
        logger.error(traceback.format_exc())
        raise
