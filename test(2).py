from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb
#import matplotlib.pyplot as plt

#where train Loader used to be
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

 # load files
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt( "mnist_test.csv" , delimiter=",")

fac = 0.99 / 255 #convert grayscale values to values between 0.1 and 1, no 0 because they can cause weights not to update
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

b = torch.from_numpy(train_imgs)
print(b.size())
