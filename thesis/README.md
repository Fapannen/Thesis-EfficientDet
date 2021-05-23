# Thesis-EfficientDet

Prerequisites: git, opencv-python, python3.7, python3.7-pip (upgrade if necessary `python3.7 -m pip install --upgrade pip`. If you encounter some issues, this might solve some.)

To get started:
 
1) Clone the repository `git clone https://github.com/Fapannen/Thesis-EfficientDet.git` (not necessary)
2) Update and fetch EfficientDet repository with `git submodule update --init --recursive` (not necessary)
3) Go to automl/efficientdet directory
4) python3.7 -m pip install numpy
5) python3.7 -m pip install -r requirements.txt (will install tensorflow 2.5 if no tensorflow is installed on the system, different than what is used in the work but that should not cause any trouble. The command will result in an error with `pycocotools` but we do not need that package. The package is for offline evaluation of models, but test-dev2017 does not have public annotations and therefore offline evaluation is not possible.)
6) go to Thesis-EfficientDet/thesis directory
6) Fetch desired version of EfficientDet with `./download_model <version>`  
   (ie. "./download_model d0"). (Models downloaded in this way are not the final trained checkpoints. They reach lower scores, but in terms of an example execution, they are identical. If you insist on using the best-performing models, download them from official EfficientDet repository.)
7) Work with the model  
    * To run example demo, run `./run_model_example.sh <version> <path_to_image>`  
        To run original example, set `path_to_image` to `images/img.png`  
        ie. `./run_model_example d0 images/img.png`.
        An output image can be inspected after the script is finished in `<version>/0.jpg`
    * To convert model to tflite format, run `./convert.sh <version>`  
      If you are working with EfficientDet versions d1 and higher, please follow the following steps.   
      Omitting these steps results in a conversion error, in my case "'tf.StridedSlice' is neither a custom op or flex op".  
      * Open ../automl/efficientdet/inference.py
      * Change line 626 (in function `export`) to  
      `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]`
      * Run convert.sh script  
    * To run inference with converted model, enter the model directory and run  
        `python3.7 run_tflite.py <version> <path_to_image>` in version directory. For example, if you are in "d0" directory, run `python3.7 run_tflite.py d0 img.png`. The resulting detections can be inspected in a file "out.jpg"  
        
# Run on device

1) Convert your desired version of EfficientDet (via steps above) to .tflite file and copy this file to your target device.
2) Setup (Cross-)Compile environment
   (In my case by sourcing a configuration file)
   (Skip if building for general-purpose desktop computer)
3) Build Tensorflow Lite static library `libtensorflow-lite.a` with Cross Compile settings
   * (I have provided one I built for my desktop computer, but I am not sure whether it will work for other computers, too. I recommend trying with prebuilt one and otherwise building your own)
   * Clone tensorflow repository, e.g. `https://github.com/tensorflow/tensorflow.git`
   * Install bazel build tool (https://docs.bazel.build/versions/master/install-ubuntu.html)
   * Install flatbuffers (https://stackoverflow.com/questions/55394537/how-to-install-flatc-and-flatbuffers-on-linux-ubuntu)
   * Makefile is located at `<tensorflow_root>/tensorflow/lite/tools/make`.
   * **Run `download_dependencies.sh` before proceeding!**
   * In `<tensorflow_root>` directory, run `./tensorflow/lite/tools/make/build_lib.sh` (Different script if building for different architecture.)
   * `libtensorflow-lite.a` will be located at `<tensorflow_root>/tensorflow/lite/tools/make/gen/<ARCH>/lib`  
   * You can choose to build shared library `libtensorflowlite.so` with bazel instead. Bazel is official Tensorflow building tool, therefore there should be less issues. After you build the shared library, you need to copy it to target device.  
   * See `https://www.tensorflow.org/lite/guide/build_arm#c_library`
4) Make sure the application is linked to `libtensorflow-lite.a` (or to `libtensorflowlite.so` if you built with bazel)
   * You can copy `libtensorflow-lite.a` to `Thesis-EfficientDet/thesis/c++/libs` directory.
   * By default, the application expects `libtensorflow-lite.a` file to be present in `/libs` directory. If you have built the shared library, please edit Makefile to look for `libtensorflowlite` instead of `libtensorflow-lite` 
5)  Install OpenCV library (https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
   * You don't need to build opencv-contrib.
   * After running `make install`, run `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib` so that linker can find newly built libraries.
6) `make efficientdet` or `make mobilenet` in `Thesis-EfficientDet/thesis/c++` directory Should now produce an executable binary `measure_[efficientdet | mobilenet]`.
7) Copy `measure_<model>` on your device and execute as `./measure_<model> <model_file>.tflite <images_folder> <input_image_size>` 
   * if executing on general-purpose computer, copy all necessary files to one directory to ensure the app runs smoothly and the guide remains accurace.
8) Model runs inference on all files in `<images_folder>` directory. An output log is produced in current directory.

# Evaluate the results

1) Build an output log using steps described above.
2) If you evaluated EfficientDet with input size e.g. 512, run the following `python3.7 parse_efficientdet_log.py  512`
3) If you evaluated MobileNet with input size e.g. 320, run the following `python3.7 parse_mobilenet_log.py 320` 
4) As I was testing this on my desktop computer, reading memory from `/proc/self/smaps_rollup` resulted in empty memory entries, because my OS does not support `smaps_rollup` binary, only default `smaps`. Empty memory logs result in the parsing script failing. Consider using original logs from `/results/<model>` directory. Runtime of the script is about 1-3 minutes on my computer.
5) Script produces either `json_outputs_efficientdet.json` or `json_outputs_mobilenet.json` file, depending on which evaluation has been done. This file can be zipped and sent to COCO evaluation server for evaluation. (https://competitions.codalab.org/competitions/20794#participate )
6) If you require per-class analysis, please download `scoring output log` from COCO evaluation server and save it in `/results/<model>` folder. After you have done that for all models that you wish to be evaluated, edit `models_paths` variable in `per_class_analysis.py` accordingly. This script will produce several `csv` files for you.
