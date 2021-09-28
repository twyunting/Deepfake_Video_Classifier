# Install the packages
# import cv2
import sys, os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import numpy as np
from numpy import asarray

# Read images
img_real = []
# path = '/content/drive/MyDrive/American_University/2021_Fall/DATA-793-001_Data Science Practicum/Datasets/manipulated_sequences/Deepfakes/raw/videos/data_test'
path = '../../Datasets/manipulated_sequences/Deepfakes/raw/videos/data_test'

for root, _, files in os.walk(path):
    current_directory_path = os.path.abspath(root)
    for f in files:
        name, ext = os.path.splitext(f)
        if ext == ".jpg":
            current_image_path = os.path.join(current_directory_path, f)
            current_image = imageio.imread(current_image_path)
            img_real.append(current_image)
img_real = np.array(img_real, dtype=object)
for img in img_real:
  print(img.shape)
print(len(img_real))

def meanSubtraction(arr):
  new_arr = []
  for i in range(len(arr)):
    img = arr[i]
    img = img.astype(np.float32) # convert from integers to floats
    mean = img.mean() # calculate global mean
    img = img - mean # centering of pixels
    img /= img.std()
    new_arr.append(img)
  new_arr = np.array(new_arr, dtype=object)
  return new_arr

img_real = meanSubtraction(img_real)
print(len(img_real))
print(img_real[0])

U_real = []
S_real = []
V_real = []
for i in range(2):
  [U, S, V] = np.linalg.svd(img_real[i])
  U_real.append(U)
  S_real.append(S)
  V_real.append(V)
U_real = np.array(U_real)
S_real = np.array(S_real)
V_real = np.array(V_real)
print(U_real.shape)

