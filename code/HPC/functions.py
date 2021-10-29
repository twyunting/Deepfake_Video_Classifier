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


#emptyArray = []
#path = "../data/data_fake"

def readAllImages(emptyArray, path):
    for root, _, files in os.walk(path):
        current_directory_path = os.path.abspath(root)
        for f in files:
            name, ext = os.path.splitext(f)
            if ext == ".jpg":
                current_image_path = os.path.join(current_directory_path, f)
                current_image = cv2.imread(current_image_path)
                img_fake.append(current_image)
    emptyArray = np.array(img_fake, dtype=object)
    return emptyArray


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
        #img = [np.round(img, 2) for i in range(len(arr))]
        new_arr.append(img)
    new_arr = np.array(new_arr, dtype=object)
    return new_arr


def svdTraining(arr):
    U_arr = []
    S_arr = []
    V_arr = []
    for i in range(720):
        U, S, V = np.linalg.svd(arr[i], full_matrices=False)
        U_arr.append(U)
        S_arr.append(S)
        V_arr.append(V)
    U_arr = np.array(U_arr)
    S_arr = np.array(S_arr)
    V_arr = np.array(V_arr)
    return U_arr, S_arr, V_arr


def svdVaildation(arr):
    U_arr = []
    S_arr = []
    V_arr = []
    for i in range(720, 860, 1):
        U, S, V = np.linalg.svd(arr[i], full_matrices=False)
        U_arr.append(U)
        S_arr.append(S)
        V_arr.append(V)
    U_arr = np.array(U_arr)
    S_arr = np.array(S_arr)
    V_arr = np.array(V_arr)
    return U_arr, S_arr, V_arr

def svdTesting(arr):
    U_arr = []
    S_arr = []
    V_arr = []
    for i in range(860, 1000, 1):
        U, S, V = np.linalg.svd(arr[i], full_matrices=False)
        U_arr.append(U)
        S_arr.append(S)
        V_arr.append(V)
    U_arr = np.array(U_arr)
    S_arr = np.array(S_arr)
    V_arr = np.array(V_arr)
    return U_arr, S_arr, V_arr
