import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import logging

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.activation = activation

    def forward(self, input):
        x = self.conv2d(input)
        mask = torch.sigmoid(self.mask_conv2d(input))
        if self.activation is not None:
            x = self.activation(x)
        return x * mask

class GatedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeconv2d, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.mask_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.activation = activation

    def forward(self, input):
        x = self.conv2d(input)
        mask = torch.sigmoid(self.mask_conv2d(input))
        if self.activation is not None:
            x = self.activation(x)
        return x * mask

class CoarseNet(nn.Module):
    def __init__(self):
        super(CoarseNet, self).__init__()
        # Encoder
        self.conv1 = GatedConv2d(4, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = GatedConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        
        # Decoder
        self.deconv1 = GatedDeconv2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = GatedDeconv2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = GatedDeconv2d(32, 3, kernel_size=4, stride=2, padding=1, activation=None)

    def forward(self, input, mask):
        # Concatenate input and mask
        masked_input = input * (1 - mask) + mask
        x = torch.cat([masked_input, mask], dim=1)
        
        # Encoding
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoding
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        
        # Output is the inpainted image
        return torch.tanh(x)

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        # Encoder
        self.conv1 = GatedConv2d(4, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = GatedConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Attention branch
        self.attention = nn.Sequential(
            GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        
        # Decoder
        self.deconv1 = GatedDeconv2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = GatedDeconv2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = GatedDeconv2d(32, 3, kernel_size=4, stride=2, padding=1, activation=None)

    def forward(self, input, mask, coarse_output):
        # Use coarse output as the base
        x = torch.cat([coarse_output, mask], dim=1)
        
        # Encoding
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Attention mechanism
        x = self.attention(x)
        
        # Decoding
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        
        # Output is the refined image
        refined = torch.tanh(x)
        
        # Combine with original input based on mask
        return input * (1 - mask) + refined * mask

class DeepFill(nn.Module):
    def __init__(self):
        super(DeepFill, self).__init__()
        self.coarse_net = CoarseNet()
        self.refine_net = RefineNet()
        
    def forward(self, input, mask):
        # First stage: coarse network
        coarse_output = self.coarse_net(input, mask)
        
        # Second stage: refinement network
        refined_output = self.refine_net(input, mask, coarse_output)
        
        return refined_output

def load_model():
    """Helper function to load the DeepFill model"""
    logger = logging.getLogger('app.deepfill')
    
    try:
        logger.info("Initializing DeepFill model")
        model = DeepFill()
        
        # Set to evaluation mode
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        logger.info("DeepFill model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading DeepFill model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def inpaint_with_deepfill(image, mask):
    """
    Perform inpainting using the DeepFill model.
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 255]
        mask: numpy array of shape (H, W) with values in [0, 255]
    
    Returns:
        numpy array of shape (H, W, 3) with values in [0, 255]
    """
    logger = logging.getLogger('app.deepfill')
    
    try:
        # Check if model inputs are valid
        if image is None or mask is None:
            logger.error("Input image or mask is None")
            raise ValueError("Input image or mask is None")
            
        # Ensure mask is binary
        if mask.max() > 1:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load the model
        model = load_model()
        logger.debug("Model loaded successfully")
        
        # Get original dimensions
        original_h, original_w = image.shape[:2]
        
        # Resize if needed (to reduce memory consumption)
        max_size = 512
        if original_h > max_size or original_w > max_size:
            scale = max_size / max(original_h, original_w)
            new_h, new_w = int(original_h * scale), int(original_w * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
            mask_resized = cv2.resize(mask, (new_w, new_h))
            logger.debug(f"Resized input from {original_h}x{original_w} to {new_h}x{new_w}")
        else:
            image_resized = image
            mask_resized = mask
            
        # Preprocess the image and mask
        # Convert image to float and normalize to [-1, 1]
        image_float = image_resized.astype(np.float32) / 127.5 - 1
        
        # Convert mask to float and normalize to [0, 1]
        mask_float = mask_resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        # Image: (H, W, 3) -> (1, 3, H, W)
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Mask: (H, W) -> (1, 1, H, W)
        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0).to(device)
        
        # Ensure the mask has the same dimensions as the image for the model
        mask_tensor_3c = mask_tensor.repeat(1, 3, 1, 1)
        
        # Perform inpainting
        with torch.no_grad():
            output = model(image_tensor, mask_tensor_3c)
        
        # Postprocess the output
        # Convert back to numpy and denormalize
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np + 1) * 127.5
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)
        
        # Resize back to original dimensions if needed
        if original_h > max_size or original_w > max_size:
            output_np = cv2.resize(output_np, (original_w, original_h))
            logger.debug(f"Resized output back to {original_h}x{original_w}")
        
        logger.info("Inpainting completed successfully")
        return output_np
    
    except Exception as e:
        logger.error(f"Error in inpaint_with_deepfill: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
