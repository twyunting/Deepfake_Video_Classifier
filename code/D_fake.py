# Install the packages
import cv2
import sys, os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import imageio
import numpy as np
from numpy import asarray
from scipy import linalg
import math


# Read images
img_fake = []
# path = '../data/data_fake'
path = "../data/data_fake"

for root, _, files in os.walk(path):
    current_directory_path = os.path.abspath(root)
    for f in files:
        name, ext = os.path.splitext(f)
        if ext == ".jpg":
            current_image_path = os.path.join(current_directory_path, f)
            current_image = cv2.imread(current_image_path)
            img_fake.append(current_image)
img_fake = np.array(img_fake, dtype=object)
#for img in img_fake:
  #print(img.shape)
print("Total image is {}".format(len(img_fake)))
print(img_fake)

def meanSubtraction(arr):
  new_arr = []
  for i in range(len(arr)):
    img = arr[i]
    img = img.astype(np.float64) # convert from integers to floats
    mean = img.mean() # calculate global mean
    img = img - mean # centering of pixels
    img /= img.std()
    new_arr.append(img)
  new_arr = np.array(new_arr, dtype=object)
  return new_arr

img_fake = meanSubtraction(img_fake)
# print(len(img_fake))
# print(img_fake[0])

U_fake = []
S_fake = []
V_fake = []
for i in range(len(img_fake)):
  U, S, V = np.linalg.svd(img_fake[i], full_matrices=False)
  U_fake.append(U)
  S_fake.append(S)
  V_fake.append(V)
U_fake = np.array(U_fake)
S_fake = np.array(S_fake)
V_fake = np.array(V_fake)

print("U_fake is")
print(U_fake)
print("S_fake is")
print(S_fake)
print("V_fake is")
print(V_fake)
print("One of the shape in U_fake is {}".format(U_fake[69].shape))
print("One of the shape in S_fake is {}".format(S_fake[993].shape))
print("One of the shape in V_fake is {}".format(V_fake[500].shape))
print(U_fake.dtype, U_fake[100].dtype)

"""
References:
https://stackoverflow.com/questions/7143723/applying-svd-throws-a-memory-error-instantaneously
"""
