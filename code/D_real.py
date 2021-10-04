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
import torch

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
#for img in img_real:
  #print(img.shape)
print("Total image is {}".format(len(img_real)))

def meanSubtraction(arr):
  new_arr = []
  for i in range(len(arr)):
    img = arr[i]
    img = img.astype(np.float32) # convert from integers to floats
    mean = img.mean() # calculate global mean
    img = img - mean # centering of pixels
    #img /= img.std()
    #img = [np.round(img, 2) for i in range(len(arr))]
    new_arr.append(img)
  new_arr = np.array(new_arr, dtype=object)
  return new_arr

img_real = meanSubtraction(img_real)
# print(len(img_real))
# print(img_real[0])

############################################################
def svdTraining(arr):
  U_real = []
  S_real = []
  V_real = []
  for i in range(720):
    U, S, V = np.linalg.svd(arr[i], full_matrices=False)
    U_real.append(U)
    S_real.append(S)
    V_real.append(V)
  U_real = np.array(U_real)
  S_real = np.array(S_real)
  V_real = np.array(V_real)
  return (U_real, S_real, V_real)
############################################################

U_real = []
S_real = []
V_real = []
for i in range(720):
  U, S, V = np.linalg.svd(img_real[i], full_matrices=False)
  U_real.append(U)
  S_real.append(S)
  V_real.append(V)
U_real = np.array(U_real)
S_real = np.array(S_real)
V_real = np.array(V_real)

print("U_real is")
print(U_real)
print("S_real is")
print(S_real)
print("V_real is")
print(V_real)
print("One of the shape in U_real is {}".format(U_real[69].shape))
print("One of the shape in S_real is {}".format(S_real[93].shape))
print("One of the shape in V_real is {}".format(V_real[500].shape))
print(U_real.dtype, U_real[100].dtype)

print("--------------tensor---------------------")
"""
D_real_torch = torch.from_numpy(U_real, S_real)
print("The torch shape is {}".format(D_real_torch.shape))
print(D_real_torch.shape)
"""
"""
References:
https://stackoverflow.com/questions/7143723/applying-svd-throws-a-memory-error-instantaneously
"""
