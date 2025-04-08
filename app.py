from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
import traceback
import os
import torch
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('app')

# Import the functions from separate modules
from enhance import super_resolution
from colorize import colorize_image

app = Flask(__name__)

def load_image_from_request(file):
    try:
        # Read image file
        image_data = file.read()
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        app.logger.error(f"Error in load_image_from_request: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

def image_to_base64(image):
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Save image to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        app.logger.error(f"Error in image_to_base64: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inpaint', methods=['POST'])
def inpaint():
    try:
        logger.debug("Received inpaint request")
        
        # Get image data from request
        file = request.files.get('file')
        if not file:
            logger.error("No file provided in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        # Get mask data
        mask_data = request.form.get('mask')
        if not mask_data:
            logger.error("No mask data provided in request")
            return jsonify({'error': 'No mask data provided'}), 400
        
        logger.debug("Using OpenCV inpainting")
        
        # Convert base64 mask to numpy array
        if mask_data and ',' in mask_data:
            # Extract the base64 part
            mask_data = mask_data.split(',')[1]  
            try:
                # Decode base64 mask
                mask_bytes = base64.b64decode(mask_data)
                mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
                mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    logger.error("Failed to decode mask data")
                    return jsonify({'error': 'Failed to decode mask data'}), 400
            except Exception as e:
                logger.error(f"Error decoding mask: {str(e)}")
                return jsonify({'error': f'Error decoding mask: {str(e)}'}), 400
        else:
            logger.error("Invalid mask data format")
            return jsonify({'error': 'Invalid mask data format'}), 400
        
        # Load the uploaded image
        try:
            image_bytes = file.read()
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to decode image data")
                return jsonify({'error': 'Failed to decode image data'}), 400
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400
            
        # Make sure mask is binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Debug: log image and mask dimensions
        logger.debug(f"Image shape: {image_rgb.shape}, Mask shape: {mask.shape}")
        
        # Process the image using OpenCV inpainting
        logger.info("Using OpenCV for inpainting")
        result = cv2.inpaint(image_rgb, mask, 3, cv2.INPAINT_TELEA)
        
        # Convert the result to PIL Image for saving
        result_pil = Image.fromarray(result)
        
        # Save to a BytesIO object
        img_io = io.BytesIO()
        result_pil.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Return the image directly instead of base64
        logger.debug("Returning inpainted image")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error in inpaint route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/super_resolution', methods=['POST'])
def enhance():
    try:
        logger.debug("Super resolution request received")
        if 'image' not in request.files:
            logger.error("No image in request")
            return jsonify({'error': 'No image provided'}), 400
        
        logger.debug("Loading image")
        # Load image from request
        file = request.files['image']
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image data")
            return jsonify({'error': 'Failed to decode image data'}), 400
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.debug(f"Loaded image shape: {image_rgb.shape}")
        
        logger.debug("Performing super resolution")
        try:
            # Perform super resolution
            result_rgb = super_resolution(image_rgb)
        except Exception as e:
            logger.error(f"Super resolution failed: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
        # Convert back to BGR for OpenCV encoding
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        logger.debug("Converting result to base64")
        # Convert result to base64
        _, buffer = cv2.imencode('.png', result_bgr)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'result': result_base64})
    
    except Exception as e:
        logger.error(f"Error in enhance route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/colorize', methods=['POST'])
def colorize():
    try:
        logger.debug("Colorization request received")
        if 'image' not in request.files:
            logger.error("No image in request")
            return jsonify({'error': 'No image provided'}), 400
        
        logger.debug("Loading image")
        # Load image from request
        file = request.files['image']
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image data")
            return jsonify({'error': 'Failed to decode image data'}), 400
        
        logger.debug("Performing colorization")
        try:
            # Perform colorization with fixed render factor
            result = colorize_image(image, render_factor=35)
        except Exception as e:
            logger.error(f"Colorization failed: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
        # Create an image buffer for the response
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'result': result_base64})
    
    except Exception as e:
        logger.error(f"Error in colorize route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
