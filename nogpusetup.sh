pip install gdown


gdown '19EIcT9ei0QBGWSppdr0JgyK5vQv3GvtQ'
gdown '1VXEucnpOw_uM-tczoQUO9VoLipvBeiCZ'
gdown '1PTciOpFiJcHng-WwnUiENttvTbYjtdR1'
gdown '1UK7U5wHRZwyD4b4zaV5_bbXJI6OtvBts'


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





