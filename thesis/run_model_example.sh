# Run example inference on example image
# Again, version of efficientDet is expected as command line argument (ei "d0")
# In addition, provide path for image
rm -rf ${1}/saved_model

python3.7 ../automl/efficientdet/model_inspect.py --runmode=saved_model \
  --model_name=efficientdet-${1} --ckpt_path=${1}/efficientdet-${1} \
  --hparams="image_size=512x512" \
  --saved_model_dir=${1}/saved_model

python3.7 ../automl/efficientdet/model_inspect.py --runmode=saved_model_infer \
  --model_name=efficientdet-${1}  \
  --saved_model_dir=${1}/saved_model  \
  --input_image=${2} --output_image_dir=${1}
