# PowerShell equivalent of download.sh
# Create necessary directories
New-Item -Path ".\experiments\pretrained_models" -ItemType Directory -Force

Write-Host "Downloading DAT model..."
# Run gdown to download the file
python -m gdown "https://drive.google.com/uc?id=1-EN4WjNNchplkE1Su0q-WcD2pRXIWm_o" -O ".\Inpainting\DAT\experiments\pretrained_models\"
