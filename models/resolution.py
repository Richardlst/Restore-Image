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

def enhance_detail_without_distortion(image, strength=0.7):
    """Enhance image details without distorting the original appearance
    
    Args:
        image: Input image
        strength: Enhancement strength (0.2-0.9 recommended)
    
    Returns:
        Enhanced image with better details but preserved structure
    """
    try:
        # Make sure we have a proper image
        if len(image.shape) < 2:
            return image
            
        # First apply bilateral filter for noise reduction while preserving edges
        if len(image.shape) == 3:  # Color image
            smooth = cv2.bilateralFilter(image, 5, 75, 75)
        else:  # Grayscale
            smooth = cv2.bilateralFilter(image, 5, 75, 75)
            
        # Create detail layer (high frequency)
        detail = cv2.subtract(image, smooth)
        
        # Enhance details by multiplying with a factor
        enhanced_detail = cv2.multiply(detail, np.ones_like(detail) * (1.0 + strength))
        
        # Add enhanced details back to smooth image
        result = cv2.add(smooth, enhanced_detail)
        
        # Ensure output is within valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logging.error(f"Error in enhance_detail_without_distortion: {str(e)}")
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
        
        # STEP 1: Apply adaptive noise reduction
        logger.debug("Applying adaptive noise reduction")
        
        if noise_level > 25:  # High noise
            if noise_type == 'salt_pepper':
                # Use median filter for salt & pepper noise
                logger.debug("Using median filter for high salt & pepper noise")
                if len(img_array.shape) == 3:  # Color image
                    result = img_array.copy()
                    for i in range(3):  # Process each channel
                        result[:,:,i] = cv2.medianBlur(img_array[:,:,i], 5)
                else:
                    result = cv2.medianBlur(img_array, 5)
                # Apply min-max filter to catch any remaining outliers
                result = min_max_filter(result, kernel_size=3)
            elif noise_type == 'gaussian':
                # Use bilateral filter for gaussian noise (preserves edges better)
                logger.debug("Using bilateral filter for high gaussian noise")
                result = cv2.bilateralFilter(img_array, 9, 75, 75)
            else:
                # Use NLM for unknown or mixed noise (best quality but slower)
                logger.debug("Using NLM filter for high unknown/mixed noise")
                h_value = 15  # Higher h for stronger noise reduction
                if len(img_array.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(img_array, None, h_value, h_value, 7, 21)
                else:
                    result = cv2.fastNlMeansDenoising(img_array, None, h_value, 7, 21)
        elif noise_level > 10:  # Medium noise
            if noise_type == 'salt_pepper':
                # Use median filter with smaller kernel
                logger.debug("Using median filter for medium salt & pepper noise")
                if len(img_array.shape) == 3:
                    result = img_array.copy()
                    for i in range(3):
                        result[:,:,i] = cv2.medianBlur(img_array[:,:,i], 3)
                else:
                    result = cv2.medianBlur(img_array, 3)
            elif noise_type == 'gaussian':
                # Use lighter bilateral filter
                logger.debug("Using light bilateral filter for medium gaussian noise")
                result = cv2.bilateralFilter(img_array, 7, 50, 50)
            else:
                # Use lighter NLM
                logger.debug("Using light NLM filter for medium unknown/mixed noise")
                h_value = 10
                if len(img_array.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(img_array, None, h_value, h_value, 5, 15)
                else:
                    result = cv2.fastNlMeansDenoising(img_array, None, h_value, 5, 15)
        else:  # Low or no noise - minimal processing
            logger.debug("Low noise detected, applying minimal noise reduction")
            if len(img_array.shape) == 3:
                result = cv2.bilateralFilter(img_array, 5, 35, 35)
            else:
                result = cv2.bilateralFilter(img_array, 5, 35, 35)
        
        # STEP 2: Enhance details without distortion
        logger.debug("Enhancing details without distortion")
        # Adjust enhancement strength based on noise level
        if noise_level > 25:
            detail_strength = 0.35  # Very gentle for high noise
        elif noise_level > 10:
            detail_strength = 0.5   # Moderate for medium noise
        else:
            detail_strength = 0.7   # Stronger for clean images
        
        result = enhance_detail_without_distortion(result, strength=detail_strength)
        
        # STEP 3: Apply gentle edge enhancement
        logger.debug("Applying gentle edge enhancement")
        
        if noise_level > 25:  # High noise
            # Skip this step for very noisy images
            pass
        elif noise_level > 10:  # Medium noise
            # Very gentle unsharp mask
            logger.debug("Using very gentle unsharp mask")
            result = unsharp_mask(result, radius=1, amount=0.5, threshold=15)
        else:  # Low noise
            # Gentle unsharp mask
            logger.debug("Using gentle unsharp mask")
            result = unsharp_mask(result, radius=1, amount=0.8, threshold=10)
        
        # STEP 4: Preserve original colors and appearance with careful blending
        logger.debug("Preserving original appearance with careful blending")
        
        # Calculate blend factor based on noise level
        # Higher noise = more weight to processed image
        # Lower noise = more weight to original to preserve appearance
        if noise_level > 25:
            blend_factor = 0.8  # More weight to processed for high noise
        elif noise_level > 10:
            blend_factor = 0.7  # Balanced for medium noise
        else:
            blend_factor = 0.6  # More weight to original for low noise
            
        # Blend with original
        result = cv2.addWeighted(result, blend_factor, original, 1.0 - blend_factor, 0)
        
        # Ensure result is valid
        result = np.clip(result, 0, 255).astype(np.uint8)
        
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