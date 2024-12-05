pip install gdown

gdown 1TrWkNWCQVCCvN2iQQ_3QOSGEY4FToY1h
gdown 1H5-Ci1wpJOyJC0T27dVLSZVk7jbrjoyu
gdown 1O4ipOucBCBz_-NrMXHF4OlvPdoocj6AI
gdown 1er0v8s6AQvkVBVjN9LippGqlR_6js4pV

mv *.weights weights/
mv *.conv.* weights/

cd darknet/

sed -i 's/OPENCV=1/OPENCV=0/' Makefile
sed -i 's/GPU=1/GPU=0/' Makefile
sed -i 's/CUDNN=1/CUDNN=0/' Makefile
sed -i 's/CUDNN_HALF=1/CUDNN_HALF=0/' Makefile
make
cp darknet ../
cd ..