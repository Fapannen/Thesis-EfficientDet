import numpy as np
import tensorflow as tf
import sys

version = sys.argv[1]

print("Working with EfficientDet version: ", version)

# Load saved model file
model = tf.keras.models.load_model('../' + version)
