import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16,MobileNet
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

inputShape=(224,224)
preprocess=imagenet_utils.preprocess_input

model=VGG16(weights="imagenet")

model.summary()

