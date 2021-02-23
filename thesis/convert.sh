rm -rf ${1}/saved_model
python3 ../automl/efficientdet/model_inspect.py --runmode=saved_model --model_name=efficientdet-${1} \
  --ckpt_path=${1}/efficientdet-${1} --saved_model_dir=${1}/saved_model \
  --tflite_path=${1}/efficientdet-${1}.tflite
