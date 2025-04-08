import cv2
import numpy as np
import logging
import traceback

logger = logging.getLogger('image_enhancement')

def edge_preserving_filter(image, alpha=0.2, k=2):
    """
    Apply edge-preserving filter using weighted average filtering.
    """
    filtered = image.copy().astype(np.float32)
    for _ in range(k):
        filtered = cv2.GaussianBlur(filtered, (3, 3), alpha)
    return filtered

def enhance_details(image, alpha=0.2, k=2, s=0.2):
    """
    Enhance color image using edge-preserving mask and detail enhancement.
    """
    try:
        # Ensure image is in the correct format
        if image is None:
            raise ValueError("Input image is None")
            
        # Convert to float32 for processing
        image = np.clip(image.astype(np.float32), 0, 255)
        
        # Create CLAHE object with reduced clip limit
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Reduced clip limit
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel with reduced contrast
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
        
        # Process each channel
        enhanced_channels = []
        for i in range(3):
            channel = image[:, :, i]
            
            # Apply edge-preserving filter
            smooth = edge_preserving_filter(channel, alpha, k)
            detail = channel - smooth
            
            # Calculate local statistics
            local_mean = cv2.blur(detail, (20, 20))
            local_var = cv2.blur(detail**2, (20, 20)) - local_mean**2
            local_var = np.maximum(local_var, 0)  # Ensure non-negative
            threshold = 2.0 * np.sqrt(local_var)
            
            # Enhance details with reduced intensity
            detail_mask = np.abs(detail - local_mean) >= threshold
            enhanced = channel.copy()
            enhanced[detail_mask] += s * detail[detail_mask]
            
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255)
            enhanced_channels.append(enhanced.astype(np.uint8))
        
        # Merge channels
        result = cv2.merge(enhanced_channels)
        
        # Final adjustment with reduced brightness
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=0)  # Removed brightness boost
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhance_details: {str(e)}")
        logger.error(traceback.format_exc())
        # Return original image if enhancement fails
        return image.astype(np.uint8) if image is not None else None

def apply_enhancement_pipeline(image):
    """
    Apply the complete image enhancement pipeline.
    """
    try:
        if image is None:
            raise ValueError("Input image is None")
            
        # Convert to BGR if in RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image.copy()

        # Apply enhancement
        enhanced = enhance_details(image_bgr)
        
        # Convert back to RGB
        if enhanced is not None:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            return enhanced_rgb
        else:
            return image
            
    except Exception as e:
        logger.error(f"Error in apply_enhancement_pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return image  # Return original image if enhancement fails
