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

void printVector(const std::vector<float>& v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    std::cout << v[i] << ", ";
  }

  std::cout << std::endl;
}

std::vector<std::vector<float>> getOutputVectors(const TfLiteTensor* tensor_ptr,const int num_outputs,const int output_size)
{
  std::vector<std::vector<float>> outputs;

  float* output = reinterpret_cast<float*>(tensor_ptr->data.raw);

  for (int i = 0; i < num_outputs; ++i)
  {
    std::vector<float> outvec;

    for (int j = 0; j < output_size; ++j)
    {
      outvec.push_back(output[(i * output_size) + j]);
    }

    outputs.push_back(outvec);
  }

  return outputs;
}

void drawBoundingBoxes(const std::vector<std::vector<float>>& outputs, cv::Mat& image)
{
  std::vector<std::vector<float>> boxesToDraw;

  // First output will always be drawn
  boxesToDraw.push_back(outputs[0]);

  for(size_t i = 1; i < outputs.size() - 1; i++){
    if(outputs[i] != outputs[i-1]){
      boxesToDraw.push_back(outputs[i]);
    }
  }

  for(std::vector<float>& vec : boxesToDraw){
    int imgNum = vec[0];
    int ymin   = vec[1];
    int xmin   = vec[2];
    int ymax   = vec[3];
    int xmax   = vec[4];
    int score  = vec[5];
    int label  = vec[6];

    cv::Point topRight(xmin, ymin);
    cv::Point botLeft(xmax, ymax);

    cv::rectangle(image, topRight, botLeft, cv::Scalar(0, 255, 0));
  }

  cv::imwrite("out.png", image);
}

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <path to image>\n");
    return 1;
  }

  const char* filename  = argv[1];
  const char* imagepath = argv[2];

  cv::Mat img;
  cv::Mat resized_img;

  // Open image
  // Opening using imread will have continuous memory
  img = cv::imread(imagepath, cv::IMREAD_COLOR);
  if (img.empty()){
      fprintf(stderr, "Failed to read image ...\n");
      return -1;
  }

  // Resize input image to fit the model
  cv::resize(img, resized_img, cv::Size(MODEL_RES, MODEL_RES), 0, 0, cv::INTER_CUBIC);

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  fprintf(stderr, "Number of tensors: %ld \n", interpreter->tensors_size());

  fprintf(stderr, "Input tensor name: %s\n", interpreter->GetInputName(0));
  fprintf(stderr, "Output tensor 0 name: %s\n", interpreter->GetOutputName(0));
  fprintf(stderr, "Output tensor 1154 name: %s\n", interpreter->GetOutputName(1154));

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

  printf("Just finished ...\n");

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  return 0;
}