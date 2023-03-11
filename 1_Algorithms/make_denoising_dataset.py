# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:32:35 2018

@author: root
"""

import numpy as np
import cv2
import os
import random
# import matplotlib.pyplot as plt
import zipfile
import os


imageSize = 128
m = (0,0,0)
noise_factor = 0.1

source_path = '/project/ctb-akhanf/myousif9/Neural_Networks_project'
# saveDir = os.path.join(source_path, "output/Train/denoising")
imagePath = os.path.join(source_path, "output/Train/resampled.zip")
outZip = os.path.join(source_path, "output/Train/denoising.zip")

# imagePath = '/home/alexander/Desktop/brightness_per/ILSVRC2014_128X128/'

# trainDir = os.listdir(imagePath+'train')
# testDir = os.listdir(imagePath+'test')

# x_train = np.zeros((len(trainDir),imageSize,imageSize,3))
#ytrain = np.zeros((len(trainDir),imageSize,imageSize,3))

# New zip file
zipf = zipfile.ZipFile(outZip, 'w', zipfile.ZIP_DEFLATED)

with zipfile.ZipFile(imagePath, mode="r") as archive:
    n_files = len(archive.namelist())
    x_train = np.zeros((n_files,128,128,3))
    for filename in archive.namelist():
        data = archive.read(filename)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        image = image + noise_factor * np.random.normal(loc=0.0, scale=255, size=x_train.shape[1:])
        retval, buf = cv2.imencode('.jpeg', image)
        zipf.writestr(filename, buf)
zipf.close()
        

# trainDir = os.listdir(imagePath+'train')
    
# for i in range(len(trainDir)):
    
#     # s = (random.randint(0,100),random.randint(0,100),random.randint(0,100))
#     tempImg = cv2.imread(imagePath+'train/'+trainDir[i])
#     #y_test[i,:,:,:] = tempImg    
#     x_train[i,:,:,:] =  tempImg + noise_factor * np.random.normal(loc=0.0, scale=255, size=x_train.shape[1:])


# x_train = np.clip(x_train, 0., 255.)
# np.save(saveDir+'/x_train', x_train)
# #np.save(saveDir+'/y_test', y_test)

# del x_train#,y_train


'''
x_test = np.zeros((len(testDir),imageSize,imageSize,3))
#y_test = np.zeros((len(testDir),imageSize,imageSize,3))

print("Train ready!!!")

testDir = os.listdir(imagePath+'test')
    
for i in range(len(testDir)):
    
    s = (random.randint(0,100),random.randint(0,100),random.randint(0,100))
    tempImg = cv2.imread(imagePath+'test/'+testDir[i])
    #y_test[i,:,:,:] = tempImg    
    x_test[i,:,:,:] =  tempImg + noise_factor * np.random.normal(loc=0.0, scale=255, size=x_test.shape[1:])


x_test = np.clip(x_test, 0., 255.)
np.save(saveDir+'/x_test', x_test)
#np.save(saveDir+'/y_test', y_test)

del x_test#,y_test
'''    
print("All images were read!!!")  
