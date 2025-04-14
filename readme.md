# YOLOv8 Tank Detection: Auto-Labeling and Training Workflow

This project outlines the workflow for using **GroundingDINO** (via `autodistill`) to automatically generate **bounding box** labels for an image dataset containing "tank" objects, followed by splitting the data and training a **YOLOv8 object detection** model using the auto-labeled dataset.

## Table of Contents

1.  [System Requirements](#system-requirements)
2.  [Environment Setup](#environment-setup)
3.  [Step 1: Auto-Labeling (Bounding Box Generation)](#step-1-auto-labeling-bounding-box-generation)
4.  [Step 2: Splitting the Dataset (Train/Valid)](#step-2-splitting-the-dataset-trainvalid)
5.  [Step 3: Training YOLOv8](#step-3-training-yolov8)
6.  [Next Steps and Notes](#next-steps-and-notes)
7.  [Final Directory Structure](#final-directory-structure)

## System Requirements

*   Python 3.10+
*   `pip` and `virtualenv` (or `python -m venv`)
*   `git`
*   **NVIDIA GPU:** Highly recommended (potentially necessary for good performance). This workflow was tested with an NVIDIA P40 (24GB VRAM). Appropriate NVIDIA drivers and CUDA toolkit must be installed.
*   Internet connection (to download libraries and model weights).
*   Directory containing the original images to be labeled.

## Environment Setup

1.  **Open Terminal:** Navigate to your project directory (e.g., `/root/src/`).
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
    *(Your command prompt should now start with `(venv)`)*
4.  **Install necessary libraries:**
    *Note: Due to previously encountered version compatibility issues, we will install in a specific order and specify some versions.*

    a.  Install `wheel` (required for building some packages):
        ```bash
        pip install wheel
        ```
    b.  Install compatible versions of `transformers` and `peft` (identified during debugging):
        ```bash
        pip install transformers==4.35.2 peft==0.9.0
        ```
        *(If you encounter issues later, you might need to try slightly different versions, but this is the starting point that resolved previous errors)*
    c.  Install the remaining main libraries, including PyTorch with the appropriate CUDA version (e.g., `cu118`):
        ```bash
        pip install autodistill-grounding-dino supervision opencv-python numpy torch torchvision torchaudio PyYAML tqdm ultralytics --index-url https://download.pytorch.org/whl/cu118
        ```
        *(Important: Change `cu118` to the CUDA version matching your system if needed - e.g., `cu117`, `cu121`. Check the PyTorch website if unsure.)*

## Step 1: Auto-Labeling (Bounding Box Generation)

Use the `auto_labeling.py` script (the version using only GroundingDINO) to generate bounding box labels.

1.  **Preparation:**
    *   Ensure you have a directory containing the original images (e.g., `/root/src/datasets/`).
    *   Ensure you have the `auto_labeling.py` script (provided and corrected previously).
2.  **Run the command:**
    ```bash
    # Make sure venv is activated
    python auto_labeling.py \
        --input_dir datasets \
        --output_dir my_dataset \
        --ontology "tank:tank" \
        --box_threshold 0.45 \
        --nms_threshold 0.5
    ```
    *   `--input_dir datasets`: **Important:** Path (relative or absolute) to the directory containing original images. This example assumes `datasets` is in `/root/src`. Adjust as needed.
    *   `--output_dir my_dataset_bbox`: The directory that will be created to store the results (images, labels, yaml).
    *   `--ontology "tank:tank"`: Defines that the prompt "tank" should be assigned the class label "tank".
    *   `--box_threshold 0.45`: Confidence threshold for GroundingDINO.
    *   `--nms_threshold 0.5`: Threshold for Non-Maximum Suppression to remove duplicate bounding boxes.
3.  **Result:** The script will create the `my_dataset` directory with the following structure:
    ```
    my_dataset_bbox/
    ├── images/         # Contains copies of the original images
    │   ├── image1.jpg
    │   └── ...
    ├── labels/         # Contains the .txt label files (bounding boxes)
    │   ├── image1.txt
    │   └── ...
    └── data.yaml       # Dataset configuration file
    ```
    *Each `.txt` file will contain lines in the format: `0 <x_center> <y_center> <width> <height>` (where `0` is the index for the `tank` class).*

## Step 2: Splitting the Dataset (Train/Valid)

Use the `split_dataset.py` script to divide the labeled data into training and validation sets.

1.  **Preparation:** Ensure you have the `split_dataset.py` script (provided previously).
2.  **Run the command:**
    ```bash
    # Make sure venv is activated
    python split_dataset.py --input_dataset_dir my_dataset_bbox --train_ratio 0.8
    ```
    *   `--input_dataset_dir my_dataset_bbox`: **Important:** This must be the output directory from Step 1.
    *   `--train_ratio 0.8`: Splits 80% of the data for the training set, 20% for the validation set.
3.  **Result:** Files within `my_dataset_bbox/images/` and `my_dataset_bbox/labels/` will be moved into `train/` and `valid/` subdirectories. The `my_dataset_bbox/data.yaml` file will also be updated with the correct paths.

## Step 3: Training YOLOv8

Use the `train.py` script to train the YOLOv8 object detection model.

1.  **Preparation:** Ensure you have the `train.py` script (provided and corrected previously).
2.  **Run the command (Example optimized for P40):**
    ```bash
    # Make sure venv is activated
    python train_yolov8.py \
        --data my_dataset/data.yaml \
        --model yolov8s.pt \
        --epochs 300 \
        --imgsz 640 \
        --batch -1 \
        --patience 50 \
        --workers 8 \
        --device 0 \
        --project runs/tank_detection_optimal \
        --name my_model
    ```
    *   `--data my_dataset/data.yaml`: **Important:** Path to the `data.yaml` file updated in Step 2. Ensure this path is correct from where you run the command.
    *   `--model yolov8s.pt`: Selects the largest model for higher accuracy potential.
    *   `--epochs 300`: Number of training epochs.
    *   `--batch -1`: Lets ultralytics auto-select the optimal batch size for VRAM.
    *   `--patience 100`: Enables early stopping if no improvement after 100 epochs.
    *   `--workers 8`: Number of data loading threads.
    *   `--device 0`: Specifies using the first GPU (P40).
    *   `--project runs/tank_detection_optimal`: Main directory to save results.
    *   `--name my_model`: Specific subdirectory name for this run.
3.  **Result:** The training process will begin. Results (best weights `best.pt`, last weights `last.pt`, logs, graphs, etc.) will be saved in the `runs/tank_detection_optimal/run_x_optimal_e200` directory.

## Next Steps and Notes

*   **VERIFY LABELS:** Auto-labeling is not perfect. **It is highly recommended to spend time reviewing and correcting the generated label files in `my_dataset_bbox/labels/train` and `my_dataset_bbox/labels/valid`** using annotation tools like Roboflow, CVAT, or Label Studio before serious training.
*   **Hyperparameters:** Experiment with different values for `--epochs`, `--batch`, `--imgsz`, `--model`, learning rate (`lr0`), etc., to find the best configuration for your specific dataset.
*   **GroundingDINO Prompt:** If the initial auto-labeling results are poor, try changing the prompt in the `--ontology` argument of the `auto_labeling_dino.py` script (e.g., `"military tank":tank`).
*   **Inference:** Use the `best.pt` weights file generated after training to perform predictions on new images.

## Final Directory Structure (After completing all steps)
