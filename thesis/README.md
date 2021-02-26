# Thesis-EfficientDet

To get started:

1) Clone the repository
2) Update and fetch EfficientDet repository with `git submodule update --init --recursive`
3) Fetch desired version of EfficientDet with `./download_model <version>`  
   (ie. "./download_model d0")
4) Work with the model  
    * To run example demo, run `./run_model_example.sh <version> <path_to_image>`  
        To run original example, set `path_to_image` to `images/img.png`  
        ie. `./run_model_example d0 images/img.png`  
    * To convert model to tflite format, run `./convert.sh <version>`  
    * To run inference with converted model, run  
        `python3 run_tflite.py <version> <path_to_image>` 
