# DeepLearning_ComputerVision
Deep Learning for Computer Vision

## 1. Automatic_Image_Captioning

This project implements a CNN-RNN model for automatic image captioning. The model takes an image as input and generates a sequence of text that describes the image content. 

**Key features:**

* Uses a pre-trained ResNet model as the CNN backbone.
* Employs a LSTM network as the RNN for sequence generation.
* Trained on the COCO dataset.

**Usage:**

1. Install the required dependencies (e.g., TensorFlow, Keras, OpenCV).
2. Download the pre-trained weights.
3. Run the `Image_Captioning.ipynb` script to generate captions for images.

## 2. Human Intrusion Detection with Real-time Tracking

This project implements a real-time human intrusion detection system using a YOLOv3 deep learning model. It utilizes OpenCV for video processing and object tracking. Key functionalities include:

* Human Detection: Detects humans within an image/video stream.
Object Tracking: Tracks the detected humans using a Euclidean distance tracker.
* Real-time Intrusion Detection: Defines a Region of Interest (ROI) and triggers an alert if a human enters the ROI.
* Data Recording: Records human trajectories including bounding box coordinates and frame numbers for further analysis (optional).

**Features:**

* Utilizes YOLOv3 model for efficient human detection.
* Employs Euclidean distance tracker for robust human tracking.
* Supports real-time video processing with ROI definition.
* Generates human trajectory data (optional).

**Requirements:**

1. Python 3.x
1. OpenCV
1. NumPy
1. Tensorflow/Keras (for custom model usage)
1. YOLOv3 pre-trained weights and configuration files

**Usage:**

1. Install the required libraries.
2. Download the YOLOv3 pre-trained weights and configuration files (coco.names, yolov3-320.cfg, yolov3-320.weights).
3. Define the ROI coordinates in the code (refPt variable).
4. Run the script: python human_intrusion_detection.py

**Note:**

1. This project can be extended to support additional object classes by modifying the required_class_index list and potentially retraining the YOLOv3 model.
2. The script currently saves human trajectories to a CSV file ("Trajectory.csv"). This functionality can be disabled by commenting out the relevant lines.
3. This project provides a starting point for building a real-time human intrusion detection system with tracking capabilities.