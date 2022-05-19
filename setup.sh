#!/bin/bash -i

pip install gdown
pip install -r requirements.txt
pip install git+https://github.com/ozgur/python-firebase

gdown '19EIcT9ei0QBGWSppdr0JgyK5vQv3GvtQ'
gdown '1VXEucnpOw_uM-tczoQUO9VoLipvBeiCZ'
gdown '1PTciOpFiJcHng-WwnUiENttvTbYjtdR1'
gdown '1UK7U5wHRZwyD4b4zaV5_bbXJI6OtvBts'


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




