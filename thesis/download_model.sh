# Download model as described in 3)
# https://github.com/google/automl/tree/master/efficientdet

# Script expects argument which version of efficientDet to download (ie "d0")

mkdir ${1}
cd ${1}

# Download and unpack the model
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-${1}.tar.gz
wget https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png -O img.png

tar xf efficientdet-${1}.tar.gz

cp ../run_tflite.py .
cp ../coco_label_map.py .
