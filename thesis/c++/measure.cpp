#include <cstdio>
#include <cstdint> // uint8_t
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
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

// Set which model to work with
int MODEL_RES = D0;
int CHANNELS  = 3;

// An application to measure <everything>.
//
// Usage: measure <model> <input_image>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <path to image>\n");
    return 1;
  }

  const char* filename  = argv[1];
  const char* imagepath = argv[2];

  cv::Mat img;

  // Open image
  // Opening using imread will have continuous memory
  img = cv::imread(imagepath, cv::IMREAD_COLOR);
  if (img.empty()){
      fprintf(stderr, "Failed to read image ...\n");
      return -1;
  }

  // Resize input image to fit the model
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(MODEL_RES, MODEL_RES), 0, 0, cv::INTER_CUBIC);

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
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
  memcpy((void*)input, (void*) resized_img.data, MODEL_RES * MODEL_RES * CHANNELS * sizeof(uint8_t));

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");

  // Output tensor is "float32 [1, 100, 7]"
  TfLiteTensor* outtensor = interpreter->output_tensor(0);

  auto outputs = getOutputVectors(outtensor, 100, 7);

  drawBoundingBoxes(outputs, resized_img);

  printf("Done\n");

  return 0;
}