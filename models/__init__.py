# Models package
# This file is required to make Python treat the directory as a package

# Import necessary modules
import os
import sys
import logging

# Get logger
logger = logging.getLogger('app.models')

# Import models if possible
try:
    from . import deepfill
    logger.info("DeepFill model imported successfully")
    DEEPFILL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import DeepFill model: {str(e)}")
    DEEPFILL_AVAILABLE = False
