#!/bin/bash

# ---------------------------------------------
# YOLOv8 Tank Detection - Setup Environment
# ---------------------------------------------
set -e

PROJECT_DIR=$(pwd)
echo "🚀 Bắt đầu cài đặt môi trường tại: $PROJECT_DIR"

# Step 1: Create virtual environment
echo "🔧 Tạo virtual environment..."
python3 -m venv venv

# Step 2: Activate virtual environment
echo "✅ Kích hoạt môi trường..."
source venv/bin/activate

# Step 3: Upgrade pip and install wheel
echo "📦 Cập nhật pip và cài đặt wheel..."
pip install --upgrade pip
pip install wheel

# Step 4: Install transformers and peft
echo "📚 Cài transformers==4.35.2 và peft==0.9.0..."
pip install transformers==4.35.2 peft==0.9.0

# Step 5: Install autodistill core
echo "📦 Cài autodistill core từ PyPI..."
pip install autodistill

# Step 6: Cài autodistill-grounding-dino từ GitHub
echo "🔧 Clone và cài autodistill-grounding-dino từ GitHub..."
git clone https://github.com/autodistill/autodistill-grounding-dino.git
cd autodistill-grounding-dino
pip install -e .
cd ..

# Step 7: Install the rest libraries (supervision, opencv, torch, ultralytics,...)
echo "📦 Cài đặt supervision, opencv, torch, ultralytics..."
pip install supervision opencv-python numpy torch torchvision torchaudio PyYAML tqdm ultralytics --index-url https://download.pytorch.org/whl/cu118

echo "✅ Hoàn tất! Bạn có thể kích hoạt môi trường bằng: source venv/bin/activate"
