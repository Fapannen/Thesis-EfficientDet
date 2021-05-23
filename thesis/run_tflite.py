# Script taken from official TFlite documentation available at
# tensorflow.org/lite/guide/inference

import numpy as np
import tensorflow as tf
import cv2
import sys
import os

# Different versions of EfficientDet need different inputs
# Map the versions to desired inputs

resolutions = {
  "d0" : 512,
  "d1" : 640,
  "d2" : 768,
  "d3" : 896,
  "d4" : 1024,
  "d5" : 1280,
  "d6" : 1280,
  "d7" : 1536,
  "lite0" : 320,
  "lite2" : 448,
  "lite3" : 512
}

def draw_boxes(image, predictions):
    color = (255, 0, 0)
    thickness = 2
    
    for box in predictions:
        cv2.rectangle(image, (box[2], box[1]), (box[4], box[3]), color, thickness)

def show_image(image):
    cv2.imshow("output", image)
    cv2.waitKey()

version = sys.argv[1]

model = "efficientdet-" + version + ".tflite"

interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
print(input_shape)
print(output_shape)
input_img = cv2.imread(sys.argv[2])
input_img = cv2.resize(input_img, (resolutions[version], resolutions[version]))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
interpreter.set_tensor(input_details[0]['index'], [input_img])

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

scored_predictions = [box for box in output_data[0] if box[5] > 0]

draw_boxes(input_img, scored_predictions)
input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
show_image(input_img)
