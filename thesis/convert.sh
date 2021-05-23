rm -rf ${1}/saved_model
python3.7 ../automl/efficientdet/model_inspect.py --runmode=saved_model --model_name=efficientdet-${1} \
  --ckpt_path=${1}/efficientdet-${1} --saved_model_dir=${1}/saved_model \
  --min_score_thresh=0.0 \
  --tflite_path=${1}/efficientdet-${1}.tflite
