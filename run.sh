#!/bin/bash

# ---------------------------------------------
# YOLOv8 Tank Detection - Full Pipeline Runner
# ---------------------------------------------
set -e

# Activate virtual environment
echo "✅ Kích hoạt môi trường ảo..."
source venv/bin/activate

# Define paths and parameters (sửa nếu bạn đổi cấu trúc)
INPUT_DIR="datasets"
OUTPUT_DIR="my_dataset_bbox"
ONTOLOGY="tank:tank"
BOX_THRESHOLD=0.45
NMS_THRESHOLD=0.5
TRAIN_RATIO=0.8
DATA_YAML="$OUTPUT_DIR/data.yaml"
MODEL_NAME="yolov8s.pt"
EPOCHS=300
IMG_SIZE=640
BATCH=-1
PATIENCE=50
WORKERS=8
DEVICE=0
PROJECT="runs/tank_detection_optimal"
NAME="my_model"

# Step 1: Auto-labeling
echo "📦 Bước 1: Auto-labeling với GroundingDINO..."
python auto_labeling.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ontology "$ONTOLOGY" \
    --box_threshold "$BOX_THRESHOLD" \
    --nms_threshold "$NMS_THRESHOLD"

# Step 2: Split dataset
echo "✂️ Bước 2: Chia tập dữ liệu (train/valid)..."
python split_dataset.py \
    --input_dataset_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO"

# Step 3: Train YOLOv8
echo "🎯 Bước 3: Huấn luyện mô hình YOLOv8..."
python train_yolov8.py \
    --data "$DATA_YAML" \
    --model "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --imgsz "$IMG_SIZE" \
    --batch "$BATCH" \
    --patience "$PATIENCE" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --name "$NAME"

echo "🏁 Pipeline hoàn tất! Mô hình đã được huấn luyện và lưu tại: $PROJECT/$NAME"
