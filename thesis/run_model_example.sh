# Run example inference on example image
# Again, version of efficientDet is expected as command line argument (ei "d0")
# In addition, provide path for image

python3 ../automl/efficientdet/model_inspect.py --runmode=saved_model \
  --model_name=efficientdet-${1} --ckpt_path=${1}/efficientdet-${1} \
  --hparams="image_size=1920x1280" \
  --saved_model_dir=${1}/saved_model

python3 ../automl/efficientdet/model_inspect.py --runmode=saved_model_infer \
  --model_name=efficientdet-${1}  \
  --saved_model_dir=${1}/saved_model  \
  --input_image=${2} --output_image_dir=${1}
