# Thesis-EfficientDet

To get started:

1) Clone the repository
2) Update and fetch EfficientDet repository with "git submodule update --init --recursive"
3) Fetch desired version of EfficientDet with "./download_model <version>"
   (ie. "./download_model d0")
4) Work with the model
4.1) To run example demo, run "./run_model_example.sh <version>"
4.2) To convert model to tflite, run "./convert.sh <version>"
4.3) To run inference with converted tflite model, run "python3 run.py <path_to_image>"
