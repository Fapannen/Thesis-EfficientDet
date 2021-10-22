#include <cstdio>
#include <cstdint> // uint8_t
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem> // inference all images in a file.
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "utils.hpp"

// Set what resolution to work with
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
  
  TfLiteTensor* inTensor   = interpreter->input_tensor(0);
  TfLiteTensor* outBoxTensor  = interpreter->output_tensor(0);
  TfLiteTensor* outClassTensor  = interpreter->output_tensor(1);
  TfLiteTensor* outScoreTensor  = interpreter->output_tensor(2);
  TfLiteTensor* outNumDetTensor  = interpreter->output_tensor(3);

  uint8_t* input = reinterpret_cast<uint8_t*>(inTensor->data.raw);

  cv::Mat img;
  cv::Mat resizedImg;

  std::vector<std::vector<float>> boxes;
  std::vector<std::vector<float>> classes;
  std::vector<std::vector<float>> scores;
  std::vector<std::vector<float>> numdet;

  int imgCnt = 0;

  // Evaluate on all images in provided directory
  for (const auto & entry : std::filesystem::directory_iterator(imageDir))
  {
    img = readImage(entry.path(), MODEL_RES, MODEL_RES);

    memcpy((void*)input, (void*) img.data, MODEL_RES * MODEL_RES * CHANNELS * sizeof(uint8_t));

    logMemoryUsage(logging);

    auto inferenceTimeDuration = timedInference(interpreter.get());

    std::vector<std::vector<float>> outputs;

    boxes   = getOutputVectors(outBoxTensor, 100, 4);
    classes = getOutputVectors(outClassTensor, 100, 1);
    scores  = getOutputVectors(outScoreTensor, 100, 1);
    numdet  = getOutputVectors(outClassTensor, 1, 1);

    for (int i = 0; i < 100; ++i)
    {
      std::vector<float> out;
      out.push_back(numdet[0][0]); 
    
      for (auto& f : boxes[i])
      {
        out.push_back(f);
      }
  
      for(auto& f : scores[i]){
          out.push_back(f);
      }

      for (auto& f : classes[i])
      {
          out.push_back(f + 1);
      }

      outputs.push_back(out);
    }
    
    log<std::string>(logging, "Image file", std::string(entry.path()));
    log<int>(logging, "Inference time in ms", static_cast<int>(inferenceTimeDuration.count()));
    logOutputs<float>(logging, outputs);
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