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

1. Change `ARCH` variable according to your GPU in `Makefile`, if you have a different GPU.
2. `chmod +x setup.sh`  
3. `./setup.sh`  

The setup script will install the required packages and download the weights from Google Drive for YOLOv4. It will also make necessary changes to some flags in the Makefile and then compile. If you don't have a GPU, use the `nogpu_localsetup.sh` script instead.

## Run Project

`python wrong_side_driving_detection.py --show_frames 1`  

Usage:   

```wrong_side_driving_detection.py [-h] [--input INPUT]
                                       [--youtube_link YOUTUBE_LINK]
                                       [--input_mode INPUT_MODE]
                                       [--profile PROFILE]
                                       [--save_output_video SAVE_OUTPUT_VIDEO]
                                       [--show_frames SHOW_FRAMES]
                                       [--use_firebase USE_FIREBASE]
```
optional arguments:  
  `-h, --help`            
  show this help message and exit  
  `--input INPUT`         
  input video file path, default:
                        demo_data/thai_cctv.mp4  
  `--youtube_link YOUTUBE_LINK`  
                        input video youtube link, default:
                        https://www.youtube.com/watch?v=ATq6ZbRQtDY  
  `--input_mode INPUT_MODE`  
                        Mode of input, yt for youtube, fl for file, default:
                        yt  
  `--profile PROFILE`     
  Perform Python profiling to analyse bottlenecks,
                        default: 0  
  `--save_output_video` SAVE_OUTPUT_VIDEO
                        Save output video file, default: 1  
  `--show_frames SHOW_FRAMES`
                        Show output frames as detection is taking place,
                        default: 0  
  `--use_firebase USE_FIREBASE`
                        Use your firebase db to store violation images, make
                        sure to create sensitive_data.json in the same
                        directory as this program, default: 0

## Publication

[IJTES, Volume 2, Issue 3, July-2022](https://mapscipub.com/download/2192/)

## Contributions

Open source contributions are welcome! Please submit a pull request to the [GitHub repository](https://github.com/sriramcu/yolov4_wrong_side_driving_detection).
