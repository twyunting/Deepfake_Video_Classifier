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
from numpy import ndarray

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

# Training 720 videos. That is, 720x7 images
def meanSubtraction(arr):
  new_arr = []
  for i in range(len(arr)):
    img = arr[i]
    img = np.array(img, dtype=np.float32) # convert from integers to floats
    #img = img.astype(np.float32)
    mean = img.mean() # calculate global mean
    img = img - mean # centering of pixels
    #img /= img.std()
    img = [np.round(img, 2) for i in range(len(arr))]
    new_arr.append(img)
  new_arr = np.array(new_arr, dtype=object)
  return new_arr

img_fake = meanSubtraction(img_fake)
#print(len(img_fake))
#print(img_fake[0])


U_fake = []
S_fake = []
V_fake = []
"""
U_fake = np.array([], dtype=np.float32)
S_fake = np.array([], dtype=np.float32)
V_fake = np.array([], dtype=np.float32)
"""

for i in range(10):
  U, S, V = np.linalg.svd(img_fake[i], full_matrices=True)
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
print("One of the shape in U_fake is {}".format(U_fake[0].shape))
print("One of the shape in S_fake is {}".format(S_fake[1].shape))
print("One of the shape in V_fake is {}".format(V_fake[1].shape))
print(U_fake.dtype, U_fake[2].dtype)


"""
References:
https://stackoverflow.com/questions/7143723/applying-svd-throws-a-memory-error-instantaneously
"""
