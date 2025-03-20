import cv2
from PIL import Image
import numpy as np
from enhance import super_resolution
import matplotlib.pyplot as plt

def test_enhancement(image_path):
    # Read image
    original = Image.open(image_path)
    
    # Apply super resolution
    print("Applying super resolution...")
    enhanced = super_resolution(original)
    
    # Convert enhanced result to PIL Image for saving
    enhanced_img = Image.fromarray(enhanced)
    
    # Save the result
    output_path = 'enhanced_output.jpg'
    enhanced_img.save(output_path)
    print(f"Enhanced image saved as: {output_path}")
    
    # Display original and enhanced images side by side
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced)
    plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.savefig('comparison.jpg')
    print("Comparison saved as: comparison.jpg")

if __name__ == "__main__":
    # Test with sample image
    test_enhancement("tiger.jpg")
