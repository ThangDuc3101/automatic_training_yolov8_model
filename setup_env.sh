#!/bin/bash
sudo apt update
sudo apt install python3.10-venv -y

# Create Python virtual environment
python3 -m venv autodistill_env
source autodistill_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install autodistill core and plugins
pip install autodistill autodistill-grounded-sam autodistill-yolov8 roboflow
pip install scikit-learn

# Install PyTorch (CUDA 11.8) - bạn có thể đổi link nếu dùng CPU hoặc CUDA khác
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "✅ All dependencies are ready."
echo "👉 To activate environment: source autodistill_env/bin/activate"

