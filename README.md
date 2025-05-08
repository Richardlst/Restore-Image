# Image Restoration Web Application

This web application provides two main features:
1. Image Inpainting - Restore missing or damaged parts of images
2. Super Resolution - Enhance image quality and make images sharper

## Project Structure

- `gradio_app.py` - Main application and route handlers
- `restore.py` - Contains the image inpainting functionality 
- `enhance.py` - Contains the super resolution functionality
- `models/` - Contains AI model related code
- `templates/` - Contains HTML templates

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python gradio_app.py
```
Or use the provided batch file:
```bash
run.bat
```

3. Open your web browser and go to `http://localhost:5000`

## How to Use

### Image Inpainting
1. Upload an image using the file input in the Inpainting section
2. Draw on the areas you want to restore (these areas will be marked in black)
3. Click "Process" to restore the image

### Super Resolution
1. Upload an image using the file input in the Super Resolution section
2. Click "Enhance" to improve the image quality

## Note
The current implementation uses basic image processing techniques. For better results, you can integrate more advanced AI models like:
- For inpainting: LaMa, DeepFillv2
- For super resolution: ESRGAN, Real-ESRGAN

## Development
The application has been structured to separate the two main functionalities (image restoration and enhancement) into different modules to make maintenance and extension easier.
