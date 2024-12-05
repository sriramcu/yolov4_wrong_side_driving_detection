# Wrong Side Driving Detection using YOLOv4

A Computer Vision (YOLOv4) based project to autonomously detect and penalise vehicles driving on the wrong side of the road.  


## Run Project in Colab

### Training (Optional)

The model has already been trained and the inference notebook uses the weights we stored in Drive from the training phase. If you want to train it yourself, then visit the following colab notebook to train the model:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sriramcu/yolov4_wrong_side_driving_detection/blob/master/notebooks/Training.ipynb)

### Inference
Visit the following colab notebook to view a demo of our project:  
  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sriramcu/yolov4_wrong_side_driving_detection/blob/master/notebooks/Yolov4_WrongSideDetection.ipynb)


## Run Project Locally

### Prerequisites

**Local machine OS must be Linux**. Additionally, the local setup has been tested on the following:

1. Ubuntu 20.04
2. Python 3.9
3. CUDA 12.0
4. CUDNN 9.3
5. Tensorflow 2.18.0
6. NVIDIA GeForce RTX 3060 Laptop GPU - thus Makefile contains the corresponding line: `ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]`

### Setup

`chmod +x setup.sh`  
`./setup.sh`  

## Run Project

`python wrong_side_driving_detection.py --show_frames 1`  

## Publication

[IJTES, Volume 2, Issue 3, July-2022](https://mapscipub.com/download/2192/)

## Contributions

Open source contributions are welcome! Please submit a pull request to the [GitHub repository](https://github.com/sriramcu/yolov4_wrong_side_driving_detection).
