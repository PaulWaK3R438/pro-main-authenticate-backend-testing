# YOLO Backend Testing – Fake Face Detection System

## Overview

The YOLO Backend Testing module focuses on evaluating and validating a custom-trained YOLOv8 deep learning model for real-time fake face detection and classification. The backend integrates computer vision capabilities with Python-based inference to identify faces and classify detected instances into two categories: **Real** and **Fake**.

The system uses the Ultralytics YOLO framework, PyTorch, and OpenCV to perform object detection, process visual input, and generate predictions with associated confidence scores. The trained model is tested against webcam input and image data to assess its detection performance, inference speed, and suitability for integration into a web-based application.

## Objectives

- Validate the functionality of the trained YOLOv8 model.
- Test real-time face detection and Real/Fake classification.
- Evaluate prediction confidence scores and bounding-box localization.
- Measure inference latency and frames per second (FPS).
- Identify false-positive and false-negative predictions.
- Investigate performance under different lighting conditions, facial angles, and image qualities.
- Prepare the model for backend integration with a web application.

## Technology Stack

- **Python** – Backend logic and model execution.
- **YOLOv8 (Ultralytics)** – Object detection and classification.
- **PyTorch** – Deep learning inference and model execution.
- **OpenCV** – Webcam capture and image processing.
- **cvzone** – Visualization of bounding boxes and prediction labels.
- **Google Colab with GPU acceleration** – Model training and experimentation.

## Testing Workflow

1. Load the trained model weights (`n_version_e_300.pt`).
2. Initialize the webcam or provide image input.
3. Process incoming frames through the YOLO inference pipeline.
4. Extract predicted classes, bounding-box coordinates, and confidence scores.
5. Apply a configurable confidence threshold to filter detections.
6. Visualize predictions using color-coded bounding boxes: green for Real and red for Fake.
7. Monitor FPS and evaluate inference responsiveness.
8. Analyze detection errors and optimize the backend for future web application integration.

## Expected Outcome

The testing module is intended to establish a reliable inference pipeline that produces interpretable face-detection predictions while maintaining responsive performance. The evaluation helps identify opportunities to improve model accuracy, reduce latency, and support integration with real-time web applications.

**Note:** Real/Fake predictions represent the model's learned classifications and should not be interpreted as definitive proof of authenticity. Actual accuracy and performance must be established through systematic testing.

## Project Status

**Currently in Testing and Optimization Phase.**
