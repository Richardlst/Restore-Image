import os
import time 
import logging
import numpy as np
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt
from scipy import stats
from scipy import ndimage

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('models.resolution')

def detect_noise_type(image):
    """Detect type of noise in image
    Args:
        image: numpy array (HxWxC)
    Returns:
        str: Type of noise ('gaussian', 'salt_pepper', or 'speckle')
    """
    # Convert to grayscale if colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()

    # Check for salt and pepper noise
    black_pixels = hist[0]
    white_pixels = hist[-1]
    if black_pixels > 0.01 and white_pixels > 0.01:
        return 'salt_pepper'

    # Check distribution
    _, skewness, _ = stats.skewnorm.fit(gray.flatten())
    if abs(skewness) < 0.5:  # Close to normal distribution
        return 'gaussian'
    
    return 'speckle'

def apply_median_filter(image, kernel_size=5):
    """Apply median filter to reduce noise while preserving edges
    Args:
        image: numpy array (HxWxC)
        kernel_size: size of median filter kernel (odd number)
    Returns:
        Filtered image
    """
    # Apply median filter separately to each channel for better noise reduction
    if len(image.shape) == 3:  # Color image
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = cv2.medianBlur(image[:,:,i], kernel_size)
        return result
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=5, sigma=0):
    """Apply Gaussian filter for Gaussian noise
    Args:
        image: numpy array
        kernel_size: size of Gaussian kernel (odd number)
        sigma: standard deviation. If 0, calculated based on kernel size
    Returns:
        Filtered image
    """
    # Automatically calculate optimal sigma based on kernel size if not specified
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # Apply bilateral filter for edge preservation
    if len(image.shape) == 3:  # Color image
        return cv2.bilateralFilter(image, kernel_size, sigma*2, sigma*2)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter to preserve edges while removing noise
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_adaptive_filter(image, noise_type):
    """Apply appropriate filter based on noise type
    """
    if noise_type == 'salt_pepper':
        return apply_median_filter(image, kernel_size=5)
    elif noise_type == 'gaussian':
        return apply_bilateral_filter(image)
    else:  # speckle
        return apply_gaussian_filter(image, kernel_size=5, sigma=0)

def unsharp_mask(image, kernel_size=(3, 3), sigma=0.5, amount=3.0, threshold=5):
    """Apply unsharp masking to enhance edges
    Args:
        image: numpy array (HxWxC)
        kernel_size: size of gaussian blur kernel
        sigma: gaussian blur sigma
        amount: how much to enhance (1.0 = 100%)
        threshold: minimum brightness change
    Returns:
        Sharpened image
    """
    # Convert to float
    image = image.astype(np.float32)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Calculate unsharp mask
    sharpened = float(amount + 1) * image - float(amount) * blurred
    
    # Clip values and convert back
    sharpened = np.clip(sharpened, 0, 255)
    
    # Calculate difference for threshold
    diff = sharpened - image
    
    # Apply threshold to difference
    if threshold > 0:
        low_contrast_mask = np.absolute(diff) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened.astype(np.uint8)

def estimate_noise_level(image):
    """Estimate the noise level in the image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute noise level using Laplacian
    noise_sigma = cv2.Laplacian(gray, cv2.CV_64F).var() ** 0.5
    return noise_sigma

def apply_nlm_denoising(image, h=None):
    """Apply Non-local Means Denoising with conservative parameters
    Args:
        image: Input image
        h: Filter strength (None for auto)
    Returns:
        Denoised image
    """
    if h is None:
        noise_level = estimate_noise_level(image)
        # Giảm cường độ lọc xuống
        h = max(5, min(noise_level * 1.5, 15))
    
    if len(image.shape) == 3:
        # Giảm template window và search window size
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 5, 11)
    return cv2.fastNlMeansDenoising(image, None, h, 5, 11)

def apply_adaptive_median_filter(image, min_size=3, max_size=7):
    """Apply adaptive median filter that adjusts kernel size based on local noise
    Args:
        image: Input image
        min_size: Minimum kernel size (odd number)
        max_size: Maximum kernel size (odd number)
    Returns:
        Filtered image
    """
    result = np.copy(image)
    
    # Convert to grayscale for noise estimation if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate local variance using a sliding window
    local_var = cv2.GaussianBlur(gray, (5, 5), 0)
    local_var = cv2.Laplacian(local_var, cv2.CV_32F)
    local_var = np.abs(local_var)
    
    # Normalize variance to 0-1 range
    if local_var.max() > 0:
        local_var = local_var / local_var.max()
    
    # Apply median filter with adaptive kernel size
    if len(image.shape) == 3:  # Color image
        for i in range(3):
            channel = image[:,:,i]
            filtered = np.copy(channel)
            
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    # Skip if at the image border
                    if y < max_size//2 or y >= image.shape[0] - max_size//2 or \
                       x < max_size//2 or x >= image.shape[1] - max_size//2:
                        continue
                    
                    # Determine kernel size based on local variance
                    variance = local_var[y, x]
                    k_size = int(min_size + (max_size - min_size) * variance)
                    k_size = k_size if k_size % 2 == 1 else k_size + 1  # Ensure odd
                    
                    # Apply median filter with calculated kernel size
                    y_start = max(0, y - k_size//2)
                    y_end = min(image.shape[0], y + k_size//2 + 1)
                    x_start = max(0, x - k_size//2)
                    x_end = min(image.shape[1], x + k_size//2 + 1)
                    
                    patch = channel[y_start:y_end, x_start:x_end]
                    filtered[y, x] = np.median(patch)
            
            result[:,:,i] = filtered
    else:
        # For grayscale images
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Skip if at the image border
                if y < max_size//2 or y >= image.shape[0] - max_size//2 or \
                   x < max_size//2 or x >= image.shape[1] - max_size//2:
                    continue
                
                # Determine kernel size based on local variance
                variance = local_var[y, x]
                k_size = int(min_size + (max_size - min_size) * variance)
                k_size = k_size if k_size % 2 == 1 else k_size + 1  # Ensure odd
                
                # Apply median filter with calculated kernel size
                y_start = max(0, y - k_size//2)
                y_end = min(image.shape[0], y + k_size//2 + 1)
                x_start = max(0, x - k_size//2)
                x_end = min(image.shape[1], x + k_size//2 + 1)
                
                patch = image[y_start:y_end, x_start:x_end]
                result[y, x] = np.median(patch)
    
    return result

def advanced_denoising(image, noise_type='unknown'):
    """Apply more effective noise reduction while preserving edges
    Args:
        image: Input image
        noise_type: Type of noise
    Returns:
        Denoised image
    """
    # Step 1: Initial noise reduction based on noise type
    if noise_type == 'salt_pepper':
        # For salt & pepper, use non-local means followed by adaptive median
        initial = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) if len(image.shape) == 3 else \
                 cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        result = apply_adaptive_median_filter(initial, min_size=3, max_size=5)
    
    elif noise_type == 'gaussian':
        # For gaussian noise, use combined non-local means and bilateral
        initial = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) if len(image.shape) == 3 else \
                 cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        result = cv2.bilateralFilter(initial, 9, 75, 75)
    
    elif noise_type == 'speckle':
        # For speckle, use combination of Gaussian and bilateral
        initial = cv2.GaussianBlur(image, (5, 5), 0)
        result = cv2.bilateralFilter(initial, 9, 75, 75)
    
    else:
        # Unknown noise - use robust NLM + bilateral combination
        initial = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) if len(image.shape) == 3 else \
                 cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        result = cv2.bilateralFilter(initial, 7, 50, 50)
    
    # Step 2: Edge preservation refinement
    # Calculate edge mask to selectively apply filtering
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate edges to protect larger edge regions
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create mask where edges = 0 (black) and non-edges = 1 (white)
    mask = 1 - (edges / 255.0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) if len(image.shape) == 3 else mask
    
    # Blend filtered image with original on edges to preserve details
    blended = mask * result + (1 - mask) * image
    
    return blended.astype(np.uint8)

def highboost_filter(image, k=2.0):
    """Apply highboost filtering to enhance edges
    Args:
        image: Input image
        k: Boosting factor (k > 1.0)
    Returns:
        Edge-enhanced image
    """
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # High-pass filter (original - blurred)
    highpass = cv2.subtract(image, blurred)
    
    # Highboost filtering: original + k * highpass
    enhanced = cv2.addWeighted(image, 1.0, highpass, k, 0)
    
    # Ensure output is within valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

def laplacian_sharpening(image, strength=0.5):
    """Apply Laplacian sharpening
    Args:
        image: Input image
        strength: Strength of sharpening effect (0.0-1.0)
    Returns:
        Sharpened image
    """
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    # Create a Laplacian kernel
    kernel = np.array([[-1, -1, -1], 
                       [-1,  8, -1], 
                       [-1, -1, -1]], dtype=np.float32)
    
    # Apply Laplacian filter
    if len(image.shape) == 3:  # Color image
        laplacian = np.zeros_like(image_float)
        for i in range(3):  # Process each channel
            laplacian[:,:,i] = cv2.filter2D(image_float[:,:,i], -1, kernel)
    else:  # Grayscale image
        laplacian = cv2.filter2D(image_float, -1, kernel)
    
    # Add weighted Laplacian to original
    sharpened = image_float + strength * laplacian
    
    # Ensure output is within valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

def unsharp_mask(image, radius=5, amount=2.0, threshold=15):
    """Apply unsharp mask for sharpening
    Args:
        image: Input image
        radius: Blur radius
        amount: Sharpening strength
        threshold: Threshold for applying sharpening
    Returns:
        Sharpened image
    """
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    
    # Calculate the unsharp mask (original - blurred)
    unsharp_mask = cv2.subtract(image, blurred)
    
    # Apply threshold to avoid sharpening noise
    # Only sharpen pixels where difference exceeds threshold
    if threshold > 0:
        _, thresholded = cv2.threshold(
            cv2.convertScaleAbs(unsharp_mask), 
            threshold, 
            255, 
            cv2.THRESH_TOZERO
        )
        unsharp_mask = thresholded
    
    # Add weighted mask to original image
    sharpened = cv2.addWeighted(image, 1.0, unsharp_mask, amount, 0)
    
    # Ensure output is within valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened

def min_max_filter(image, kernel_size=3):
    """Apply min-max filtering to completely remove salt and pepper noise
    Args:
        image: Input image
        kernel_size: Size of kernel for min/max operations
    Returns:
        Filtered image without salt and pepper noise
    """
    # Create structuring element for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # For color images, process each channel separately
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            # Min filter removes salt (white) noise
            min_filtered = cv2.erode(image[:,:,i], kernel)
            
            # Max filter removes pepper (black) noise
            result[:,:,i] = cv2.dilate(min_filtered, kernel)
    else:
        # Min filter removes salt (white) noise
        min_filtered = cv2.erode(image, kernel)
        
        # Max filter removes pepper (black) noise
        result = cv2.dilate(min_filtered, kernel)
    
    return result

def adaptive_detail_enhancement(image, strength=1.0):
    """Enhance image details adaptively based on frequency content
    
    Args:
        image: Input image
        strength: Enhancement strength (0.5-2.0)
    
    Returns:
        Detail enhanced image
    """
    try:
        # Split image channels if color image
        if len(image.shape) == 3:
            # Convert to YUV color space to separate luminance
            image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            y_channel = image_yuv[:,:,0]
            
            # Apply enhancement only to luminance channel
            result_y = np.zeros_like(y_channel)
            
            # Multi-scale decomposition (3 scales)
            # Scale 1 - fine details
            blur1 = cv2.GaussianBlur(y_channel, (0, 0), 1.0)
            detail1 = y_channel - blur1
            
            # Scale 2 - medium details
            blur2 = cv2.GaussianBlur(y_channel, (0, 0), 3.0)
            detail2 = blur1 - blur2
            
            # Scale 3 - coarse details
            blur3 = cv2.GaussianBlur(y_channel, (0, 0), 9.0)
            detail3 = blur2 - blur3
            
            # Base layer (very low frequency)
            base = blur3
            
            # Enhance each detail layer with different weights
            fine_weight = min(2.0, max(0.8, strength * 1.5))  # Fine details get stronger enhancement
            medium_weight = min(1.7, max(0.7, strength * 1.2))  # Medium details
            coarse_weight = min(1.4, max(0.6, strength))  # Coarse details get lighter enhancement
            
            # Reconstruct enhanced image by adding weighted details to base
            result_y = base + (detail1 * fine_weight) + (detail2 * medium_weight) + (detail3 * coarse_weight)
            
            # Clip to valid range
            result_y = np.clip(result_y, 0, 255).astype(np.uint8)
            
            # Merge back with color channels
            image_yuv[:,:,0] = result_y
            result = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
            
        else:
            # For grayscale images, similar process without color space conversion
            # Multi-scale decomposition
            blur1 = cv2.GaussianBlur(image, (0, 0), 1.0)
            detail1 = image - blur1
            
            blur2 = cv2.GaussianBlur(image, (0, 0), 3.0)
            detail2 = blur1 - blur2
            
            blur3 = cv2.GaussianBlur(image, (0, 0), 9.0)
            detail3 = blur2 - blur3
            
            base = blur3
            
            # Enhance each detail layer
            fine_weight = min(2.0, max(0.8, strength * 1.5))
            medium_weight = min(1.7, max(0.7, strength * 1.2))
            coarse_weight = min(1.4, max(0.6, strength))
            
            # Reconstruct
            result = base + (detail1 * fine_weight) + (detail2 * medium_weight) + (detail3 * coarse_weight)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
        return result
    
    except Exception as e:
        logging.error(f"Error in adaptive detail enhancement: {str(e)}")
        return image

def texture_preserving_filter(image, strength=0.4):
    """Apply texture-preserving filtering to enhance image quality while reducing noise
    
    Args:
        image: Input image
        strength: Filter strength (0.0-1.0)
    
    Returns:
        Filtered image
    """
    try:
        # Convert strength to appropriate filter parameters
        sigma_s = 60 * (1 - strength)  # Spatial sigma (smaller = stronger effect)
        sigma_r = 0.4 * (1 - strength)  # Range sigma (smaller = stronger effect)
        
        # Apply edge-preserving detail filter (similar to bilateral but better preserves textures)
        # Using guided filter because it's more efficient than bilateral for texture preservation
        if len(image.shape) == 3:  # Color image
            result = np.zeros_like(image)
            
            # Process each channel
            for i in range(3):
                # Get channel
                channel = image[:,:,i]
                
                # Apply guided filter using the channel itself as guide
                radius = int(3 * (1 + strength * 2))  # Radius based on strength
                epsilon = (0.1 + strength * 0.3) ** 2  # Regularization term
                
                filtered = cv2.ximgproc.guidedFilter(
                    guide=channel,
                    src=channel,
                    radius=radius,
                    eps=epsilon
                )
                
                # Apply result to channel
                result[:,:,i] = filtered
                
        else:  # Grayscale
            radius = int(3 * (1 + strength * 2))
            epsilon = (0.1 + strength * 0.3) ** 2
            
            result = cv2.ximgproc.guidedFilter(
                guide=image,
                src=image,
                radius=radius,
                eps=epsilon
            )
            
        # Additional local contrast enhancement
        local_contrast = cv2.addWeighted(
            image, 1 + strength * 0.5,
            result, -(strength * 0.5),
            0
        )
        
        # Blend between filtered and contrast-enhanced
        alpha = 0.7
        result = cv2.addWeighted(result, alpha, local_contrast, 1-alpha, 0)
        
        return result.astype(np.uint8)
    
    except Exception as e:
        logging.error(f"Error in texture preserving filter: {str(e)}")
        return image

def has_face(image):
    """Detect if image contains a face
    
    Args:
        image: Input image
    
    Returns:
        Boolean indicating if face is detected
    """
    try:
        # Load pre-trained face cascade if available
        face_cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         'haarcascade_frontalface_default.xml')
        
        # Use default OpenCV path if file not found
        if not os.path.exists(face_cascade_path):
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Load the face cascade
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Return True if any faces found
        return len(faces) > 0
    
    except Exception as e:
        logging.error(f"Error in face detection: {str(e)}")
        return False

def skin_smoothing(image, strength=0.5):
    """Apply skin smoothing filter (commonly used in beauty apps)
    
    Args:
        image: Input image
        strength: Smoothing strength (0.0-1.0)
    
    Returns:
        Smoothed image
    """
    try:
        # Check if image is color (can't do skin smoothing on grayscale)
        if len(image.shape) < 3:
            return image
            
        # Convert to YCrCb color space which is better for skin detection
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Define skin color range in YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # Apply morphological operations to improve mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Create normalized alpha mask
        skin_alpha = skin_mask.astype(float) / 255.0
        
        # Adjust strength
        skin_alpha = skin_alpha * strength
        
        # Create 3-channel skin alpha mask
        skin_alpha_3c = np.zeros_like(image, dtype=float)
        for i in range(3):
            skin_alpha_3c[:,:,i] = skin_alpha
            
        # Create smoothed version of image
        # Use bilateral filtering for edge-preserving skin smoothing
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Additional smoothing with guided filter
        radius = int(5 + 10 * strength)
        eps = 0.1 * 0.1
        for i in range(3):
            smoothed[:,:,i] = cv2.ximgproc.guidedFilter(
                guide=image[:,:,i],
                src=smoothed[:,:,i],
                radius=radius,
                eps=eps
            )
        
        # Blend original and smoothed based on skin mask
        result = (1 - skin_alpha_3c) * image.astype(float) + skin_alpha_3c * smoothed.astype(float)
        
        return result.astype(np.uint8)
    
    except Exception as e:
        logging.error(f"Error in skin smoothing: {str(e)}")
        return image

def enhance_resolution(image):
    """Enhance image resolution and quality
    Args:
        image: PIL Image
    Returns:
        Enhanced image as numpy array
    """
    try:
        logger = logging.getLogger('models')
        logger.debug("Starting image enhancement")
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Detect noise type
        logger.debug("Detecting noise type")
        noise_type = detect_noise_type(img_array)
        logger.debug(f"Detected noise type: {noise_type}")
        
        # Estimate initial noise level
        noise_level = estimate_noise_level(img_array)
        logger.debug(f"Initial noise level: {noise_level}")
        
        # Store original for edge protection
        original = img_array.copy()
        
        # STEP 1: Always apply median filter first as priority
        logger.debug("Applying median filter as priority")
        
        # Determine kernel size based on noise level
        if noise_level > 30:
            kernel_size = 5  # Stronger filtering for high noise
        else:
            kernel_size = 3  # Lighter filtering for lower noise
        
        logger.debug(f"Using median filter with kernel size {kernel_size}")
        
        if len(img_array.shape) == 3:  # Color image
            result = img_array.copy()
            for i in range(3):  # Process each channel
                result[:,:,i] = cv2.medianBlur(img_array[:,:,i], kernel_size)
        else:  # Grayscale image
            result = cv2.medianBlur(img_array, kernel_size)
        
        # STEP 2: Apply secondary filtering based on noise type
        if noise_level > 15:  # Only apply secondary filtering for significant noise
            logger.debug("Applying secondary filtering based on noise type")
            
            if noise_type == 'salt_pepper':
                # For salt & pepper, apply min-max filter
                logger.debug("Using min-max filter for salt & pepper noise")
                min_max_size = 3 if noise_level < 30 else 5
                result = min_max_filter(result, kernel_size=min_max_size)
            elif noise_type == 'gaussian':
                # For gaussian noise, apply light bilateral filter to preserve edges
                d = 5  # Small diameter
                sigma = 30  # Conservative sigma value
                result = cv2.bilateralFilter(result, d, sigma, sigma)
                
            elif noise_type == 'speckle':
                # For speckle noise, apply light gaussian filter
                result = cv2.GaussianBlur(result, (3, 3), 0.8)
                
            else:  # unknown noise type
                # Apply light NLM denoising
                h = 10  # Light filtering strength
                if len(result.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(result, None, h, h, 5, 11)
                else:
                    result = cv2.fastNlMeansDenoising(result, None, h, 5, 11)
        
        # STEP 3: Apply texture-preserving filter
        logger.debug("Applying texture-preserving filter")
        texture_strength = 0.6 if noise_level < 15 else 0.4  # Stronger for clean images
        result = texture_preserving_filter(result, strength=texture_strength)
        
        # STEP 4: Check if image contains a face for special processing
        has_faces = has_face(result)
        logger.debug(f"Face detected: {has_faces}")
        
        if has_faces:
            # Apply skin smoothing for face images (like beauty apps)
            logger.debug("Applying smart skin smoothing")
            # Adjust smoothing strength based on noise level
            skin_strength = 0.6 if noise_level < 15 else 0.4
            result = skin_smoothing(result, strength=skin_strength)
        
        # STEP 5: Apply multi-scale detail enhancement
        logger.debug("Applying multi-scale detail enhancement")
        # Adjust enhancement strength based on noise level
        if noise_level > 25:
            detail_strength = 0.7  # Mild for high noise
        elif noise_level > 15:
            detail_strength = 1.0  # Standard for medium noise
        else:
            detail_strength = 1.3  # Stronger for clean images
            
        logger.debug(f"Using detail enhancement strength: {detail_strength}")
        result = adaptive_detail_enhancement(result, strength=detail_strength)
        
        # STEP 6: Apply final sharpening with edge-aware filter
        logger.debug("Applying edge-aware sharpening")
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) / 5.0
        
        # Calculate edge mask using Canny
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) if len(result.shape) == 3 else result
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges
        kernel_dilate = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel_dilate, iterations=1)
        
        # Create normalized edge mask (0-1)
        edge_mask = edges.astype(float) / 255.0
        
        # Apply sharpen filter
        if len(result.shape) == 3:  # Color
            sharpened = np.zeros_like(result)
            for i in range(3):
                sharpened[:,:,i] = cv2.filter2D(result[:,:,i], -1, kernel)
        else:  # Grayscale
            sharpened = cv2.filter2D(result, -1, kernel)
        
        # Only apply sharpening to edge areas
        if len(result.shape) == 3:
            edge_mask_3c = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)
            result = (edge_mask_3c * sharpened) + ((1 - edge_mask_3c) * result)
        else:
            result = (edge_mask * sharpened) + ((1 - edge_mask) * result)
        
        # Ensure output is within valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # STEP 7: Final noise suppression to remove any introduced noise
        logger.debug("Applying final noise removal")
        result = min_max_filter(result, kernel_size=3)
        
        # STEP 8: Preserve original colors and contrast
        logger.debug("Preserving original colors and contrast")
        
        # Create a blending mask to protect detailed areas
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if len(original.shape) == 3 else original
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create mask where edges = 0 (black) and non-edges = 1 (white)
        mask = edges / 255.0  # Invert the mask from previous implementation
        
        # Extend mask to 3 channels if needed
        if len(original.shape) == 3:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
        # Use more balanced blending to preserve natural look
        blend_weight = min(0.7, max(0.4, 1.0 - noise_level / 100.0))  # More balanced weights
        result = blend_weight * result + (1 - blend_weight) * original
        result = result.astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhance_resolution: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to original image if error
        if isinstance(image, Image.Image):
            return np.array(image)
        return image

def calculate_psnr(original, enhanced):
    """Calculate Peak Signal-to-Noise Ratio between original and enhanced images
    Args:
        original: Original image tensor/array
        enhanced: Enhanced image tensor/array
    Returns:
        PSNR value
    """
    try:
        return np.max(original) / np.sqrt(np.mean((original - enhanced) ** 2))
    except Exception as e:
        logger.error(f"Error calculating PSNR: {str(e)}")
        raise

def save_image(image, filename):
    """ 
    Saves unscaled Tensor Images
    image: 3D image Tensor
    filename: Name of the file to be saved
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype(np.uint8))
    image.save('%s.jpg' % filename)
    print('Saved as %s.jpg' % filename)

def plot_image(image, title=''):
    """ 
    plots the Image tensors
    image: 3D image Tensor
    title: Title for plot
    """
    image = np.asarray(image)
    image = Image.fromarray(image.astype(np.uint8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)

def downscale_image(image):
    """
    Scales down images using bicubic downsampling.
    Args:
        image: 3D or 4D tensor of preprocessed image
    """
    image_size = []
    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = np.asarray(image)

    lr_image = cv2.resize(image, (image_size[0] // 4, image_size[1] // 4), interpolation=cv2.INTER_CUBIC)

    lr_image = np.expand_dims(lr_image, 0)
    lr_image = lr_image.astype(np.float32)
    return lr_image

def main():
    url = 'https://raw.githubusercontent.com/Masterx-AI/Project_Image_Super_Resolution/main/tiger.jpg'
    r = requests.get(url, allow_redirects=True)
    open('tiger.jpg', 'wb').write(r.content)

    IMAGE_PATH = 'tiger.jpg'

    hr_image = np.array(Image.open(IMAGE_PATH))

    lr_image = downscale_image(hr_image)

    enhanced_image = enhance_resolution(lr_image)

    psnr = calculate_psnr(hr_image, enhanced_image)

    plot_image(hr_image, title="Original")
    plot_image(lr_image, title="Low Resolution")
    plot_image(enhanced_image, title="Enhanced")

    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axes = plt.subplots(1, 3)
    fig.tight_layout()
    plt.subplot(131)
    plot_image(hr_image, title="Original")
    plt.subplot(132)
    fig.tight_layout()
    plot_image(lr_image, "x4 Bicubic")
    plt.subplot(133)
    fig.tight_layout()
    plot_image(enhanced_image, "Enhanced")
    plt.savefig("ESRGAN_DIV2K.jpg", bbox_inches="tight")
    print("PSNR: %f" % psnr)

if __name__ == "__main__":
    main()