# YOLOv8 Tank Detection: Auto-Labeling and Training Workflow

This project is based on the excellent tutorial by **Roboflow** on YouTube:  
📺 [Watch the Roboflow tutorial here](https://www.youtube.com/watch?v=2OjjWh-8Iv0)

It provides a fully automated pipeline for:
- Generating bounding box annotations for "tank" objects using **GroundingDINO**
- Splitting the dataset into training/validation sets
- Training a **YOLOv8** object detection model on the auto-labeled data

---

## 🚀 System Requirements

### Software
- Python 3.10+
- `pip` and `virtualenv`
- `git`
- Linux (tested on Ubuntu 20.04+)
- Internet connection (for downloading model weights and dependencies)

### Hardware
- **NVIDIA GPU** (highly recommended for training and labeling speed)
  - CUDA-compatible GPU (e.g., NVIDIA P40 or higher)
  - CUDA and NVIDIA drivers properly installed

---

## ⚙️ Setup Instructions

> These steps will prepare your environment and install all required packages.

1. **Clone the repository** and navigate to the project directory:
    ```bash
    git clone <this-repo-url>
    cd automatic_training_yolov8_model
    ```

2. **Make the environment setup script executable and run it:**
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```

3. Once the environment is ready, activate it:
    ```bash
    source venv/bin/activate
    ```

---

## 🧠 Running the Full Training Pipeline

> This will automatically label the dataset, split it, and train YOLOv8.

1. Make the pipeline script executable:
    ```bash
    chmod +x run_pipeline.sh
    ```

2. Run it:
    ```bash
    ./run_pipeline.sh
    ```

By default, it will:
- Read your input images from `datasets/`
- Auto-label them to `my_dataset_bbox/`
- Split the dataset into training and validation
- Train a YOLOv8 model using `yolov8s.pt` and save the results to `runs/tank_detection_optimal/`

---

## 📁 Output Overview

After running the pipeline, you will see the following output structure:

```bash
my_dataset_bbox/
├── images/
│   ├── train/
│   └── valid/
├── labels/
│   ├── train/
│   └── valid/
└── data.yaml                # Dataset configuration for YOLOv8

runs/tank_detection_optimal/
└── my_model/                # Training results
    ├── weights/
    │   ├── best.pt          # Best model checkpoint
    │   └── last.pt
    ├── results.png
    ├── confusion_matrix.png
    └── other training logs...
