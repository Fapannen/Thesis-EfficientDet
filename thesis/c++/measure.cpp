#include <cstdio>
#include <cstdint> // uint8_t
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "utils.hpp"

#define D0 512
#define D1 640
#define D2 768
#define D3 896
#define D4 1024
#define D5 1280
#define D6 1280
#define D7 1536

#define LOG true

// Set which model to work with
int MODEL_RES = D0;
int CHANNELS  = 3;

// An application to measure <everything>.
//
// Usage: measure <model> <input_image>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "measure <tflite model> <path to image>\n");
    return 1;
  }

  std::ofstream logging;

  const char* modelFile = argv[1];
  const char* imageFile = argv[2];

  std::string time      = getTime();
  std::string logFile   = createLogFileName(std::string(modelFile), time);

  if(LOG){
    logging = std::ofstream(logFile);  
  }
  
  cv::Mat img;

  // Open image
  // Opening using imread will have continuous memory
  img = cv::imread(imageFile, cv::IMREAD_COLOR);
  if (img.empty()){
      fprintf(stderr, "Failed to read image ...\n");
      return -1;
  }

  // Resize input image to fit the model
  cv::Mat resizedImg;
  cv::resize(img, resizedImg, cv::Size(MODEL_RES, MODEL_RES), 0, 0, cv::INTER_CUBIC);

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(modelFile);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");

  // Input tensor is "uint8 [1, MODEL, MODEL, 3]"
  TfLiteTensor* tensor = interpreter->input_tensor(0);
  uint8_t* input = reinterpret_cast<uint8_t*>(tensor->data.raw);

  // Copy image data to input tensor
  memcpy((void*)input, (void*) resizedImg.data, MODEL_RES * MODEL_RES * CHANNELS * sizeof(uint8_t));

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");

  // Output tensor is "float32 [1, 100, 7]"
  TfLiteTensor* outtensor = interpreter->output_tensor(0);

  auto outputs = getOutputVectors(outtensor, 100, 7);

  drawBoundingBoxes(outputs, resizedImg);

  if(LOG){
    logging.close();
  }

  printf("Done\n");

  return 0;
}