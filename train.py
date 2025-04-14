# train_yolov8.py

import argparse
import os
import yaml # Vẫn giữ để đọc/ghi yaml nếu cần mở rộng, dù ultralytics tự đọc
from ultralytics import YOLO
import traceback # Để in chi tiết lỗi

def train_model(data_yaml_path, model_variant, epochs, img_size, batch_size, device, project_name, run_name, patience, workers):
    """
    Huấn luyện mô hình YOLOv8 object detection.

    Args:
        data_yaml_path (str): Đường dẫn đến file data.yaml.
        model_variant (str): Tên biến thể YOLOv8 (ví dụ: 'yolov8n.pt', 'yolov8x.pt').
        epochs (int): Số lượng epochs để huấn luyện.
        img_size (int): Kích thước ảnh đầu vào (ví dụ: 640).
        batch_size (int): Kích thước batch (-1 để tự động xác định).
        device (str): Thiết bị để huấn luyện ('cpu', '0', '0,1', ... hoặc None để tự động).
        project_name (str): Tên thư mục dự án lưu kết quả huấn luyện.
        run_name (str): Tên thư mục con cho lần chạy huấn luyện này.
        patience (int): Số epochs chờ đợi để dừng sớm nếu không có cải thiện.
        workers (int): Số luồng tải dữ liệu.
    """

    # --- Kiểm tra sự tồn tại của file data.yaml ---
    if not os.path.isfile(data_yaml_path):
        print(f"Error: data.yaml file not found at {data_yaml_path}")
        print("Please ensure the path is correct and the file exists.")
        return

    print(f"Using dataset configuration: {data_yaml_path}")
    print(f"Using model variant: {model_variant}")
    print(f"Training for {epochs} epochs with image size {img_size}, batch size {batch_size}, patience {patience}, workers {workers}.")
    print(f"Using device: {'Auto-detect' if device is None or device == '' else device}") # Sửa lại logic print device
    print(f"Results will be saved in: {project_name}/{run_name}")

    # --- Load một mô hình YOLOv8 pre-trained ---
    try:
        model = YOLO(model_variant)
    except Exception as e:
        print(f"Error loading model variant '{model_variant}': {e}")
        print("Ensure the model variant name is correct (e.g., 'yolov8n.pt', 'yolov8x.pt')")
        print("You might need an internet connection to download the weights for the first time.")
        return

    # --- Bắt đầu huấn luyện ---
    try:
        print("\nStarting training...")
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device if device else None, # Truyền None nếu device là chuỗi rỗng hoặc None
            project=project_name,
            name=run_name,
            patience=patience,
            workers=workers
            # Thêm các tham số khác ở đây nếu cần
            # optimizer='AdamW', # Ví dụ
            # lr0=0.01,        # Ví dụ
        )
        print("\nTraining finished!")
        print(f"Results, weights, and logs saved in: {results.save_dir}")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        traceback.print_exc() # In chi tiết lỗi để debug


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 object detection model.")
    parser.add_argument("--data", required=True, help="Path to the data.yaml configuration file.")
    # Đặt default là yolov8x.pt cho phần cứng mạnh
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLOv8 model variant to use (e.g., yolov8n.pt, yolov8x.pt). Default: yolov8x.pt")
    # Tăng epochs mặc định
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs. Default: 200")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (square). Default: 640")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto-batch). Default: -1")
    # Đặt default device là 0 cho GPU đầu tiên
    parser.add_argument("--device", type=str, default='0', help="Device to run on, e.g., 'cpu', '0', '0,1' or None/'' for auto-detect. Default: '0'")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory to save results. Default: runs/detect")
    parser.add_argument("--name", type=str, default="train_optimal", help="Name for the training run directory. Default: train_optimal")
    # Thêm patience và workers với giá trị mặc định tối ưu
    parser.add_argument("--patience", type=int, default=50, help="Epochs to wait for no observable improvement for early stopping. Default: 100")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for data loading. Default: 8")

    args = parser.parse_args()

    # Gọi hàm train với tất cả các tham số đã parse
    train_model(
        args.data,
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        args.device,
        args.project,
        args.name,
        args.patience,
        args.workers
    )