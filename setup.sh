#!/bin/bash -i

pip install gdown
pip install -r requirements.txt
pip install git+https://github.com/ozgur/python-firebase

gdown 1TrWkNWCQVCCvN2iQQ_3QOSGEY4FToY1h
gdown 1H5-Ci1wpJOyJC0T27dVLSZVk7jbrjoyu
gdown 1O4ipOucBCBz_-NrMXHF4OlvPdoocj6AI
gdown 1er0v8s6AQvkVBVjN9LippGqlR_6js4pV


mv *.weights weights/
mv *.conv.* weights/


cd darknet/
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
make
cd ..


nvidia-smi
/usr/local/cuda/bin/nvcc --version




