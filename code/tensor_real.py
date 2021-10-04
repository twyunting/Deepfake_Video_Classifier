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
import D_real # read D_real.py

print(torch.__version__)

path = "../data/data_real"

print("Total image is {}".format(len(img_real)))

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

img_real = meanSubtraction(img_real)
print(svdTraining(img_real))
