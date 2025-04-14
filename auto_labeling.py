# auto_label_yolov8_detection.py

import argparse
import os
import cv2
import numpy as np
import supervision as sv
# THAY ĐỔI IMPORT: Dùng GroundingDINO thay vì GroundedSAM2
from autodistill_grounding_dino import GroundingDINO # Sửa thành dòng này (1 chữ 'd') 
from autodistill.detection import CaptionOntology, DetectionBaseModel
from tqdm import tqdm
import yaml

# --- Helper Functions ---

def parse_ontology(ontology_str: str) -> dict:
    # Giữ nguyên hàm này
    ontology_dict = {}
    try:
        pairs = ontology_str.split(',')
        if not pairs or (len(pairs) == 1 and not pairs[0].strip()):
             raise ValueError("Ontology string is empty or invalid.")
        for pair in pairs:
            if ':' not in pair:
                 raise ValueError(f"Invalid pair '{pair}'. Missing ':'.")
            prompt, class_name = pair.split(':', 1)
            prompt = prompt.strip()
            class_name = class_name.strip()
            if not prompt or not class_name:
                 raise ValueError(f"Invalid pair '{pair}'. Prompt or class name is empty.")
            ontology_dict[prompt] = class_name
    except Exception as e:
        raise ValueError(f"Invalid ontology format: '{ontology_str}'. Expected 'prompt1:class1,prompt2:class2,...'. Error: {e}")
    if not ontology_dict:
         raise ValueError("Parsed ontology dictionary is empty.")
    return ontology_dict

# THAY ĐỔI HÀM: Chuyển đổi bbox sang YOLOv8 detection format
def detections_to_yolo_detection(detections: sv.Detections, class_to_index: dict, img_w: int, img_h: int) -> list:
    """
    Chuyển đổi sv.Detections (với bounding boxes) thành các dòng định dạng YOLOv8 detection.
    Format: <class_index> <x_center> <y_center> <width> <height> (normalized)
    """
    yolo_lines = []
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return yolo_lines

    for i in range(len(detections)):
        # Lấy bbox [x_min, y_min, x_max, y_max]
        box = detections.xyxy[i]
        # Lấy class name
        class_id = 0 # Index của lớp duy nhất
        try:
            # Lấy tên lớp từ class_to_index để phòng trường hợp cần dùng (ví dụ: debug)
            # Hàm items() trả về (key, value), ở đây là (name, index)
            # Tìm key (name) có value (index) là 0
            class_name = next(name for name, index in class_to_index.items() if index == class_id)
        except StopIteration:
            # Trường hợp cực kỳ hiếm gặp là class_to_index không có index 0
            print(f"Critical Error: Class index {class_id} not found in class_to_index map. Skipping detection.")
            continue

        # Chuyển đổi sang định dạng YOLO: [x_center, y_center, width, height]
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Chuẩn hóa
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm = width / img_w
        height_norm = height / img_h

        # Tạo dòng YOLO
        yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

    return yolo_lines

# --- Main Function ---

def main(args):
    try:
        ontology_dict = parse_ontology(args.ontology)
    except ValueError as e:
        print(f"Error parsing ontology: {e}")
        return

    ontology = CaptionOntology(ontology_dict)
    class_names = list(ontology.classes())

    if not class_names:
        print("Error: No classes found in the provided ontology.")
        return

    print(f"Ontology parsed. Classes: {class_names}")

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        label_dir = os.path.join(args.output_dir, "labels")
        image_out_dir = os.path.join(args.output_dir, "images")
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_out_dir, exist_ok=True)
        print(f"Output directories created/ensured at: {args.output_dir}")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return

    print("Loading GroundingDINO model...")
    try:
        # THAY ĐỔI MODEL: Load GroundingDINO
        # base_model = GroundingDINO(ontology=ontology)
        base_model = GroundingDINO(ontology=ontology, box_threshold=args.box_threshold)
        # Set threshold (kiểm tra docs xem set ở đâu, có thể trong predict)
        # Hoặc dùng NMS sau predict
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading GroundingDINO model: {e}")
        print("Please ensure you have installed the necessary dependencies:")
        print("pip install autodistill-groundingdino supervision numpy opencv-python tqdm PyYAML torch torchvision torchaudio")
        return

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    try:
        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(supported_extensions)]
        if not image_files:
            print(f"Error: No images found in input directory: {args.input_dir}")
            return
        print(f"Found {len(image_files)} images in {args.input_dir}.")
    except FileNotFoundError:
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    except Exception as e:
        print(f"Error reading input directory: {e}")
        return


    class_to_index = {name: i for i, name in enumerate(class_names)}

    print("Starting labeling process (Bounding Boxes only)...")
    for image_name in tqdm(image_files, desc="Labeling Images"):
        image_path = os.path.join(args.input_dir, image_name)
        base_name = os.path.splitext(image_name)[0]
        output_label_path = os.path.join(label_dir, f"{base_name}.txt")
        output_image_path = os.path.join(image_out_dir, image_name)

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"\nWarning: Could not read image {image_path}. Skipping.")
                continue
            h, w, _ = image.shape

            # Predict detections (chỉ có bounding box)
            # THAY ĐỔI PREDICT: Gọi predict của GroundingDINO
            # Kiểm tra cách set threshold, có thể là tham số confidence
            # detections = base_model.predict(image_path, confidence=args.box_threshold)
            detections = base_model.predict(image_path)

            # Áp dụng Non-Maximum Suppression (NMS) - tùy chọn nhưng nên làm
            if len(detections) > 0 and args.nms_threshold > 0:
                 detections = detections.with_nms(threshold=args.nms_threshold)


            if len(detections) == 0:
                with open(output_label_path, 'w') as f:
                    pass
                cv2.imwrite(output_image_path, image)
                continue

            # THAY ĐỔI CONVERT: Dùng hàm mới để tạo YOLO detection format
            yolo_lines = detections_to_yolo_detection(detections, class_to_index, w, h)

            with open(output_label_path, 'w') as f:
                f.write("\n".join(yolo_lines))

            cv2.imwrite(output_image_path, image)

        except Exception as e:
            print(f"\nError processing {image_name}: {e}")
            with open(output_label_path, 'w') as f:
                 pass

    yaml_path = os.path.join(args.output_dir, "data.yaml")
    try:
        data_yaml = {
            'path': os.path.abspath(args.output_dir),
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'names': {v: k for k, v in class_to_index.items()}
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"\nLabeling complete (Bounding Boxes). Output saved to: {args.output_dir}")
        print(f"data.yaml created at: {yaml_path}")
        print("\nNOTE: The data.yaml file assumes all images are in the 'images' folder.")
        print("You will likely need to split your data into train/val/test sets and update data.yaml accordingly.")

    except Exception as e:
        print(f"\nError creating data.yaml file: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label dataset using GroundingDINO for YOLOv8 detection format.") # Mô tả thay đổi
    parser.add_argument("--input_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled data (YOLOv8 detection format).")
    parser.add_argument("--ontology", required=True, help="Ontology string in the format '\"prompt 1\":class1,\"prompt 2\":class2,...'.")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="GroundingDINO confidence threshold for detection.")
    # Thêm tham số NMS
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="Non-Maximum Suppression (NMS) threshold (set to 0 to disable).")


    args = parser.parse_args()
    main(args)