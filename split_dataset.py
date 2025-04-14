# split_dataset.py
import os
import random
import shutil
import argparse
import yaml
from tqdm import tqdm # Thêm tqdm để có thanh tiến trình

def split_data(input_dataset_dir, train_ratio=0.8):
    """
    Chia dữ liệu trong thư mục dataset thành tập train và valid.
    input_dataset_dir: Đường dẫn đến thư mục chứa 'images' và 'labels'.
    train_ratio: Tỷ lệ dữ liệu dùng cho tập train (ví dụ: 0.8 là 80%).
    """

    images_dir = os.path.join(input_dataset_dir, "images")
    labels_dir = os.path.join(input_dataset_dir, "labels")
    yaml_path = os.path.join(input_dataset_dir, "data.yaml")

    # --- Kiểm tra sự tồn tại của thư mục và file cần thiết ---
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return
    if not os.path.isfile(yaml_path):
        print(f"Warning: data.yaml not found at {yaml_path}. It will be created, but class names might be missing if not generated previously.")
        # Có thể tạo data.yaml cơ bản nếu muốn, nhưng tốt hơn là nó đã được tạo từ script trước

    # --- Lấy danh sách tất cả các file ảnh ---
    try:
        all_images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        if not all_images:
            print(f"Error: No image files found in {images_dir}")
            return
        print(f"Found {len(all_images)} image files.")
    except Exception as e:
        print(f"Error reading images directory {images_dir}: {e}")
        return

    # --- Xáo trộn danh sách ảnh ---
    random.shuffle(all_images)

    # --- Tính toán số lượng file cho mỗi tập ---
    num_images = len(all_images)
    num_train = int(num_images * train_ratio)
    num_valid = num_images - num_train

    print(f"Splitting data: {num_train} train images, {num_valid} validation images.")

    # --- Lấy danh sách file cho từng tập ---
    train_images = all_images[:num_train]
    valid_images = all_images[num_train:]

    # --- Tạo các thư mục con cho train/valid ---
    train_img_dir = os.path.join(images_dir, "train")
    valid_img_dir = os.path.join(images_dir, "valid")
    train_lbl_dir = os.path.join(labels_dir, "train")
    valid_lbl_dir = os.path.join(labels_dir, "valid")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(valid_lbl_dir, exist_ok=True)
    print("Created train/valid subdirectories.")

    # --- Hàm di chuyển file (ảnh và label tương ứng) ---
    def move_files(file_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir, set_name):
        print(f"Moving {set_name} files...")
        for img_filename in tqdm(file_list, desc=f"Moving {set_name}"):
            base_name = os.path.splitext(img_filename)[0]
            lbl_filename = f"{base_name}.txt"

            src_img_path = os.path.join(source_img_dir, img_filename)
            src_lbl_path = os.path.join(source_lbl_dir, lbl_filename)
            dest_img_path = os.path.join(dest_img_dir, img_filename)
            dest_lbl_path = os.path.join(dest_lbl_dir, lbl_filename)

            # Di chuyển file ảnh
            if os.path.exists(src_img_path):
                try:
                    shutil.move(src_img_path, dest_img_path)
                except Exception as e:
                    print(f"\nError moving image {src_img_path} to {dest_img_path}: {e}")
            else:
                 print(f"\nWarning: Source image not found {src_img_path}. Skipping.")


            # Di chuyển file label (kiểm tra xem nó có tồn tại không)
            if os.path.exists(src_lbl_path):
                try:
                    shutil.move(src_lbl_path, dest_lbl_path)
                except Exception as e:
                    print(f"\nError moving label {src_lbl_path} to {dest_lbl_path}: {e}")
            else:
                # Không di chuyển nếu file label không tồn tại (ví dụ: ảnh không có đối tượng)
                print(f"\nWarning: Label file not found for {img_filename} ({src_lbl_path}). Skipping label move.")
                # Có thể tạo file label rỗng ở đích nếu muốn đảm bảo mọi ảnh đều có label
                # with open(dest_lbl_path, 'w') as f: pass

    # --- Di chuyển file vào thư mục train ---
    move_files(train_images, images_dir, labels_dir, train_img_dir, train_lbl_dir, "train")

    # --- Di chuyển file vào thư mục valid ---
    move_files(valid_images, images_dir, labels_dir, valid_img_dir, valid_lbl_dir, "validation")

    # --- Cập nhật file data.yaml ---
    try:
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            if data_yaml is None: # Trường hợp file rỗng
                data_yaml = {}


        # Cập nhật đường dẫn train và val (đường dẫn tương đối)
        data_yaml['train'] = 'images/train'
        data_yaml['val'] = 'images/valid'
        # Xóa 'test' nếu có hoặc để trống nếu muốn định nghĩa sau
        if 'test' in data_yaml:
            # del data_yaml['test'] # Hoặc data_yaml['test'] = ''
            data_yaml['test'] = '' # Để trống test path

        # Đảm bảo key 'path' là đường dẫn tuyệt đối đến thư mục dataset gốc
        data_yaml['path'] = os.path.abspath(input_dataset_dir)

        # Ghi lại file yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        print(f"Updated data.yaml at {yaml_path}")

    except FileNotFoundError:
         print(f"Warning: data.yaml not found at {yaml_path}. Could not update paths. Please create it manually.")
    except Exception as e:
        print(f"Error updating data.yaml: {e}")

    print("\nDataset splitting complete.")
    print(f"Dataset structure now follows YOLOv8 format inside: {input_dataset_dir}")

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split auto-labeled dataset into train/valid sets for YOLOv8 format.")
    parser.add_argument("--input_dataset_dir", required=True, help="Path to the dataset directory created by the auto-labeling script (containing images/ and labels/).")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for the training set (e.g., 0.8 for 80%).")

    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        print("Error: train_ratio must be between 0 and 1 (exclusive).")
    else:
        split_data(args.input_dataset_dir, args.train_ratio)