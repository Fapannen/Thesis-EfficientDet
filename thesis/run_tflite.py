# Script taken from official TFlite documentation available at
# tensorflow.org/lite/guide/inference

import numpy as np
import tensorflow as tf
import cv2
import sys
import os

version = sys.argv[1]

current_dir = os.getcwd()

dir = os.chdir(os.getcwd() + "/" + version)
model = "efficientdet-" + version + ".tflite"

interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_img = cv2.imread(current_dir + "/" + sys.argv[2])
input_data = cv2.resize(input_img, (512, 512))
interpreter.set_tensor(input_details[0]['index'], [input_data])

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
