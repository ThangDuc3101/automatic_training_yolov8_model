import os
import random
import shutil
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill.helpers import load_image
from tqdm import tqdm
from PIL import Image
import yaml

# Cấu hình
label = "tank"
ontology = CaptionOntology({label: label})
base_model = GroundingDINO(ontology)

image_folder = "images"
output_dir = "yolo_dataset"
train_ratio = 0.7  # 70% train, 30% valid

# Tạo thư mục output
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images", "valid"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels", "valid"), exist_ok=True)

# Load ảnh và phân chia tập
all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(all_images)
split_index = int(len(all_images) * train_ratio)
train_images = all_images[:split_index]
valid_images = all_images[split_index:]

print(f"🔍 Gán nhãn {len(all_images)} ảnh (Train: {len(train_images)}, Valid: {len(valid_images)})...")

def label_and_save(image_list, split):
    for img_name in tqdm(image_list, desc=f"Gán nhãn ({split})"):
        try:
            img_path = os.path.join(image_folder, img_name)
            image = load_image(img_path)

            detections = base_model.predict(image)

            if not detections or not isinstance(detections[0], dict):
                print(f"⚠️  Không có box cho {img_name}")
                continue

            boxes = detections[0]["boxes"]
            h, w = image.size[1], image.size[0]

            yolo_lines = []
            for box in boxes:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_lines.append(f"0 {xc} {yc} {bw} {bh}")

            # Lưu file label
            label_path = os.path.join(output_dir, "labels", split, img_name.rsplit(".", 1)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

            # Copy ảnh tương ứng
            shutil.copy(img_path, os.path.join(output_dir, "images", split, img_name))

        except Exception as e:
            print(f"❌ Lỗi khi gán nhãn cho ảnh {img_path}: {e}")

label_and_save(train_images, "train")
label_and_save(valid_images, "valid")

# Tạo file data.yaml
yaml_path = os.path.join(output_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {output_dir}\n")
    f.write("train: images/train\n")
    f.write("val: images/valid\n")
    f.write("names:\n")
    f.write(f"  0: {label}\n")

print("✅ Gán nhãn và chia tập hoàn tất!")
print(f"📄 File data.yaml đã được tạo tại: {yaml_path}")

