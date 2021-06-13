import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2


images=[]
labels=[]
path=cv2.imread("C:/Users/Galaxy/Documents/Anaconda(Spyder)/BIGDATA HANDSON/Deep Learning/fruits")