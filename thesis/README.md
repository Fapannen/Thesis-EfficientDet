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
      If you are working with EfficientDet versions d1 and higher, please follow the following steps.   
      Omitting these steps results in a conversion error, in my case "'tf.StridedSlice' is neither a custom op or flex op".  
      * Open ../automl/efficientdet/inference.py
      * Change line 604 to  
      `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]`
      * Run convert.sh script  
    * To run inference with converted model, run  
        `python3 run_tflite.py <version> <path_to_image>` in version directory  
        
# Run on device

1) Convert your desired version of EfficientDet (via steps above) to .tflite file and copy this file to your target device.
2) Setup Cross-Compile environment
   (In my case by sourcing a configuration file)
3) Build Tensorflow Lite static library `libtensorflow-lite.a` with Cross Compile settings
   * Makefile is located at `<tensorflow_root>/tensorflow/lite/tools/make`
   * **I had to disable NNAPI support!**. In Makefile, on line 268: Change `BUILD_WITH_NNAPI ?= true` to `BUILD_WITH_NNAPI ?= false`
   * run `make <CC Config>` 
   * `libtensorflow-lite.a` will be located at `<tensorflow_root>/tensorflow/lite/tools/make/gen/<ARCH>/lib`  
   * You can choose to build shared library `libtensorflowlite.so` with bazel instead. Bazel is official Tensorflow building tool, therefore there should be less issues. After you build the shared library, you need to copy it to target device. (Not tested during runtime)   
   * See `https://www.tensorflow.org/lite/guide/build_arm#c_library`
4) Make sure the application is linked to `libtensorflow-lite.a` (or to `libtensorflowlite.so` if you built with bazel)
   * You can copy `libtensorflow-lite.a` to `c++/libs` directory.
5) `make measure` Should now produce an executable binary `measure`.
6) Copy `measure` on your device and execute as `./measure <model_file>.tflite <image_input>` 
   * Model will produce `out.png` in the current directory.
