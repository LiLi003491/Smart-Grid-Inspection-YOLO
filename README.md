# Smart-Grid-Inspection-YOLO
This project implements a robust object detection system based on YOLOv5 to monitor critical equipment in power distribution rooms. It is designed to identify the status of indicators, switches, analog meters, and relay protection devices, providing a foundation for automated industrial inspection.

## Detection Categories
The model is trained to recognize and classify the following components from your dataset:

Status Indicators: Red (Close/On), Green (Open/Off), Yellow/Orange (Fault), and White (Power).

Analog Meters: Current and voltage gauges (supporting needle-position analysis).

Control Switches: Rotary switches, breakers, and position toggles.

Relay Protection Units: LCD panels and signal LEDs on protection devices (e.g., Schneider Sapam, CET).

## 🚀 Key Features
High Precision for Small Objects: Optimized for tiny targets like LED indicators in complex industrial environments.

Comprehensive Dependencies: Includes thop for FLOPs calculation, making it ideal for evaluating performance on edge computing devices (e.g., Jetson Nano, Raspberry Pi).

Visual Analytics: Utilizes seaborn and matplotlib to generate detailed training metrics and performance curves.

## 🛠️ Installation & Environment Setup

1. Clone the Repository
   
、、、Bash

git clone https://github.com/your-username/Power-Station-YOLOv5.git

cd Power-Station-YOLOv5

2. Install Dependencies

It is recommended to use a virtual environment (Conda or venv).

Bash

Install core requirements: PyTorch, OpenCV, Seaborn, etc.

pip install -r requirements.txt

## Core Requirements Highlight:

· Framework: torch>=1.7.0 (Supports GPU acceleration)

· Vision: opencv-python & Pillow

· Analytics: tensorboard, seaborn, and pandas

· Efficiency: thop (For profiling model complexity/FLOPs)

## 📂 Dataset Details
The dataset consists of high-resolution images from power station cabinets under various lighting conditions.

Format: YOLO TXT (Normalized coordinates).

Pre-processing: Contrast enhancement and color balancing were applied to handle the glare/low-light conditions common in electrical rooms.

## 🏋️ Usage
Training

Bash

python train.py --img 640 --batch 16 --epochs 100 --data ./data/power_equipment.yaml --weights yolov5s.pt

Inference (Detection)

## 🖼️ Visualization Results
The model demonstrates high reliability in complex cabinet layouts:

[Insert Image of your detection result here, e.g., runs/detect/exp/0010.jpg]

Example: Accurate detection of "Closed" (Red) and "Open" (Green) indicators on a control panel.

Run detection on new images or videos:

Bash
python detect.py --source ./data/images/test.jpg --weights runs/train/exp/weights/best.pt --c
