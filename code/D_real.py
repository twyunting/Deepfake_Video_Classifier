# Install the packages
import cv2
import sys, os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import imageio
import numpy as np
from numpy import asarray

# Read images
img_real = []
# path = '../data/data_real'
path = "../data/data_real"

for root, _, files in os.walk(path):
    current_directory_path = os.path.abspath(root)
    for f in files:
        name, ext = os.path.splitext(f)
        if ext == ".jpg":
            current_image_path = os.path.join(current_directory_path, f)
            current_image = cv2.imread(current_image_path)
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
# print(img_real[0])

U_real = []
S_real = []
V_real = []
for i in range(len(img_real)):
  [U, S, V] = np.linalg.svd(img_real[i])
  U_real.append(U)
  S_real.append(S)
  V_real.append(V)
U_real = np.array(U_real)
S_real = np.array(S_real)
V_real = np.array(V_real)

print(U_real.shape, S_real.shape, V_real.shape)

