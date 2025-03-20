import numpy as np
import cv2
from PIL import Image
import logging
import traceback
import torch
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('restore')

# Import the DeepFill model
try:
    from models.deepfill import inpaint_with_deepfill
    DEEPFILL_AVAILABLE = True
    logger.info("DeepFill model imported successfully")
except Exception as e:
    DEEPFILL_AVAILABLE = False
    logger.warning(f"DeepFill model could not be imported: {str(e)}. Falling back to OpenCV inpainting.")

def is_bgr_format(image):
    """
    Try to determine if an image is in BGR format by looking at colors
    in areas that are likely to be skin or sky
    """
    # Check if image has 3 channels (needed for color space analysis)
    if len(image.shape) != 3 or image.shape[2] != 3:
        # Can't determine format of grayscale or 4-channel images
        return False
    
    # Simple heuristic: in natural images with people or sky,
    # the blue channel usually has lower values in RGB format
    # but higher values in BGR format
    try:
        # Compute average of each channel
        avg_ch1 = np.mean(image[:, :, 0])
        avg_ch2 = np.mean(image[:, :, 1])
        avg_ch3 = np.mean(image[:, :, 2])
        
        # If the first channel has higher average than third channel,
        # it might be BGR (Blue is more prominent than Red in natural images)
        return avg_ch1 > avg_ch3
    except:
        # For safety, assume it's not BGR if we can't analyze
        return False

def inpaint_image(image, mask):
    try:
        logger.debug("Processing inpaint_image function")
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Save a copy of the original image
        original_image = image.copy()
        
        # Process mask
        if len(mask.shape) == 3:  # Color mask
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            mask_gray = mask
        
        # Apply threshold to create binary mask with a lower threshold to capture all drawn areas
        _, binary_mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up and expand the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Log mask statistics
        white_pixels = cv2.countNonZero(binary_mask)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        logger.debug(f"Mask contains {white_pixels} white pixels out of {total_pixels} ({white_pixels/total_pixels*100:.2f}%)")
        
        # If mask is too small, return original image
        if white_pixels < 10:
            logger.warning("Mask too small, returning original image")
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                return original_image
            else:
                return cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Try to use DeepFill if available
        if DEEPFILL_AVAILABLE and torch.cuda.is_available():
            try:
                logger.info("Using DeepFill model for inpainting")
                # Ensure image is in RGB format for DeepFill
                if len(original_image.shape) == 2:  # Grayscale
                    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                elif original_image.shape[2] == 4:  # RGBA
                    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
                elif original_image.shape[2] == 3:  # Assuming BGR
                    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = original_image
                
                # Call DeepFill inpainting
                result = inpaint_with_deepfill(image_rgb, binary_mask)
                return result
            except Exception as e:
                logger.error(f"DeepFill inpainting failed: {str(e)}")
                logger.error(traceback.format_exc())
                logger.info("Falling back to OpenCV inpainting")
        else:
            logger.info("DeepFill not available, using advanced OpenCV inpainting")
        
        # Prepare image for OpenCV
        if len(original_image.shape) == 2:  # Grayscale
            image_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif original_image.shape[2] == 4:  # RGBA
            image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
        elif original_image.shape[2] == 3:
            # Check if already in BGR format
            if is_bgr_format(original_image):
                image_bgr = original_image.copy()
            else:
                image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = original_image
        
        # Ensure data types are correct
        image_bgr = image_bgr.astype(np.uint8)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Perform multi-scale inpainting for better results
        # Start with a lower resolution image and mask
        height, width = image_bgr.shape[:2]
        
        # Calculate scaling factor based on image size
        # Larger images need more downscaling for reasonable processing time
        max_dimension = max(height, width)
        scale_factor = max(0.25, min(1.0, 800 / max_dimension))
        
        # Downsample for initial processing
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)
        
        # Resize image and mask to lower resolution
        small_image = cv2.resize(image_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(binary_mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        
        # Initial inpainting at lower resolution with larger radius
        initial_inpainted = cv2.inpaint(small_image, small_mask, 15, cv2.INPAINT_NS)
        
        # Progressively refine with multiple passes at different scales
        medium_scale = min(1.0, scale_factor * 2)
        if medium_scale < 1.0:
            medium_width = int(width * medium_scale)
            medium_height = int(height * medium_scale)
            
            # Upscale the initial inpainting result
            medium_inpainted = cv2.resize(initial_inpainted, (medium_width, medium_height), interpolation=cv2.INTER_LANCZOS4)
            medium_mask = cv2.resize(binary_mask, (medium_width, medium_height), interpolation=cv2.INTER_NEAREST)
            
            # Apply another pass of inpainting at medium resolution
            medium_inpainted = cv2.inpaint(medium_inpainted, medium_mask, 10, cv2.INPAINT_TELEA)
            
            # Upscale to full resolution
            full_inpainted = cv2.resize(medium_inpainted, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            # If the image is small enough, just upscale the initial inpainting directly
            full_inpainted = cv2.resize(initial_inpainted, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply final refinement pass at full resolution
        # This is more for cleaning up edges than filling large areas
        final_inpainted = cv2.inpaint(full_inpainted, binary_mask, 5, cv2.INPAINT_NS)
        
        # Create a feathered mask for blending
        # Use a large kernel for the feathering to create smoother transitions
        feathered_mask = cv2.GaussianBlur(binary_mask, (21, 21), 11) / 255.0
        feathered_mask_3d = np.stack([feathered_mask]*3, axis=-1)
        
        # Make sure original image is in BGR format for blending
        if len(original_image.shape) == 2:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif original_image.shape[2] == 4:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
        elif original_image.shape[2] == 3:
            if is_bgr_format(original_image):
                original_bgr = original_image.copy()
            else:
                original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_image.copy()
        
        # Blend original image with inpainted image using feathered mask
        # This helps create seamless transitions at the boundaries of the inpainted region
        blended = original_bgr * (1 - feathered_mask_3d) + final_inpainted * feathered_mask_3d
        
        # Ensure output is in correct format
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Convert back to RGB for consistency with other processing
        result = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        
        # Apply an additional sharpening pass to improve detail in inpainted regions
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        
        # Blend the sharpened image only in the inpainted area
        final_result = (sharpened * feathered_mask_3d + result * (1 - feathered_mask_3d)).astype(np.uint8)
        
        return final_result
    except Exception as e:
        logger.error(f"Error in inpaint_image: {str(e)}")
        logger.error(traceback.format_exc())
        
        # If we hit an error, try to return the original image in RGB format if possible
        try:
            if len(image.shape) == 2:  # Grayscale
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                # If it's already RGB, return as is
                # If it's BGR, convert to RGB
                if is_bgr_format(image):
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    return image.copy()
            else:
                return image
        except:
            # Last resort fallback
            logger.error("Couldn't return original image either, returning None")
            return None
