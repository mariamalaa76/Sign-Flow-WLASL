# WLASL Video Preprocessing & Feature Extraction

This repository contains a robust, end-to-end preprocessing pipeline for the **World-Level American Sign Language (WLASL)** dataset. The goal of this project is to transform raw video data into standardized, augmented keypoint sequences ready for training deep learning models like LSTMs or GCNs.

## 🚀 Current Project Status: Preprocessing Phase

The project currently implements a full spatial and temporal normalization pipeline using **MediaPipe Holistic**.

### Key Features Implemented:

* **Temporal Standardization:** Resamples videos of varying lengths into a fixed number of frames (e.g., 30 frames) to ensure uniform input for temporal models.
* **Anchor-Based Spatial Normalization:** Uses fixed anatomical anchors (shoulders) to normalize hand coordinates, making the data invariant to the signer's distance from the camera.
* **Dynamic Hand Bounding Boxes:** Automatically detects and scales bounding boxes around both hands with configurable padding for spatial focus.
* **Data Augmentation:** Increases dataset diversity through random rotation and scaling of normalized keypoint sequences.

---

## 🛠 Preprocessing Pipeline

### 1. Metadata Management

Loads and filters the WLASL metadata to handle video paths, frame ranges (start/end), and class labels.

### 2. Video Standardization

Each video is processed to extract 258 features per frame, covering:

* **Pose:** 132 features (x, y, z, visibility)
* **Left Hand:** 63 features (x, y, z)
* **Right Hand:** 63 features (x, y, z)

### 3. Spatial Preprocessing & Normalization

To handle different camera angles and distances, coordinates are normalized relative to a central anchor.

```python
# Example of the Bounding Box logic implemented
left, top, right, bottom = get_hand_bounding_box(hand_landmarks, frame.shape)

```

### 4. Data Augmentation

To improve model generalization, the pipeline applies:

* **Random Rotation:** ±10 degrees around the sequence center.
* **Random Scaling:** 0.9x to 1.1x scaling factors.

---

## 📈 Visualizations

The pipeline includes tools to visualize the standardized frames. Below is an example of a **75-frame original video** compressed into a **30-frame standardized sequence**:

---

## 🛠 Requirements

* Python 3.8+
* OpenCV
* MediaPipe
* NumPy & Pandas
* FFmpeg (for video playback compatibility in Colab)

## 🔜 Future Work

* [ ] Implementation of a Graph Convolutional Network (GCN) for sign classification.
* [ ] Integration of a real-time inference script using a webcam.
* [ ] Comparison between LSTM and Transformer-based architectures.

---

**Would you like me to help you write the `requirements.txt` file or the `LICENSE` file for your repository?**
