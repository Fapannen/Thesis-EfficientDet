#include <cstdio>
#include <cstdint> // uint8_t
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <experimental/filesystem> // inference all images in a file
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "utils.hpp"

// Set which resolution to work with
int MODEL_RES;
int CHANNELS  = 3;

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "measure <tflite model> <path to images folder> <input image size>\n");
    return 1;
  }

  const char* modelFile = argv[1];
  const char* imageDir  = argv[2];
  const char* modelRes  = argv[3];

  MODEL_RES = std::stoi(std::string(modelRes));

  std::string time      = getTime();
  std::string logFile   = createLogFileName(std::string(modelFile), time);

  std::chrono::time_point<std::chrono::high_resolution_clock> fullTimeStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> fullTimeEnd;
  std::chrono::time_point<std::chrono::high_resolution_clock> inferenceTimeStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> inferenceTimeEnd;

  std::ofstream logging = std::ofstream(logFile);

  log<std::string>(logging, std::string("Memory usage at the beginning (model not loaded)"));
  logMemoryUsage(logging);

  fullTimeStart = std::chrono::high_resolution_clock::now();

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(modelFile);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  interpreter->SetNumThreads(4);

  log<std::string>(logging, std::string("Memory usage after loading the model"));
  logMemoryUsage(logging);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  log<std::string>(logging, std::string("Memory usage after allocating model tensors"));
  logMemoryUsage(logging);

  log<std::string>(logging, "----------------------------------------------------------------------------");

  TfLiteTensor* inTensor                  = interpreter->input_tensor(0);

  // [1, num_boxes, 4]
  TfLiteTensor* outTensorDetectionBoxes   = interpreter->output_tensor(0);

  // [1, num_boxes]
  TfLiteTensor* outTensorDetectionClasses = interpreter->output_tensor(1);

  // [1, num_boxes]
  TfLiteTensor* outTensorDetectionScores  = interpreter->output_tensor(2);

  // [1]
  TfLiteTensor* outTensorNumBoxes         = interpreter->output_tensor(3);


  float* input = reinterpret_cast<float*>(inTensor->data.raw);

  cv::Mat img;
  cv::Mat floatImg;

  std::vector<std::vector<float>> outputDetectionBoxes;
  std::vector<std::vector<float>> outputDetectionClasses;
  std::vector<std::vector<float>> outputDetectionScores;

  int imgCnt = 0;

  int numDetections = 0;

  // Evaluate on test dataset
  for (const auto & entry : std::experimental::filesystem::directory_iterator(imageDir))
  {
    img = readImage(entry.path(), MODEL_RES, MODEL_RES);

    img.convertTo(img, CV_32FC3, 2/255.0);

    img = img - cv::Scalar(1, 1, 1);


    memcpy((void*)input, (void*) img.data, MODEL_RES * MODEL_RES * CHANNELS * sizeof(float));

    logMemoryUsage(logging);

    auto inferenceTimeDuration = timedInference(interpreter.get());


	float* numD = reinterpret_cast<float*>(outTensorNumBoxes->data.raw);
	
	numDetections = static_cast<int>(*numD);
    
    outputDetectionBoxes   = getOutputVectors(outTensorDetectionBoxes, numDetections, 4);
    outputDetectionClasses = getOutputVectors(outTensorDetectionClasses, numDetections, 1);
    outputDetectionScores  = getOutputVectors(outTensorDetectionScores, numDetections, 1);

    log<std::string>(logging, "Image file", std::string(entry.path()));
    log<int>(logging, "Inference time in ms", static_cast<int>(inferenceTimeDuration.count()));
    log<int>(logging, "Number of detections", numDetections);
    logOutputs<float>(logging, outputDetectionBoxes);
    logOutputs<float>(logging, outputDetectionClasses);
    logOutputs<float>(logging, outputDetectionScores);
    log<std::string>(logging, "----------------------------------------------------------------------------");

    std::cout << "Images processed: " << imgCnt++ << std::endl;
  }

  fullTimeEnd = std::chrono::high_resolution_clock::now();

  auto fullTimeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(fullTimeEnd - fullTimeStart);

  log<long>(logging, "Full execution time in milliseconds", static_cast<long>(fullTimeDuration.count()));
  log<float>(logging, "Full execution time in seconds", fullTimeDuration.count() / static_cast<float>(1000));

  logging.close();

  printf("Done\n");

  return 0;
}