# -*- coding: utf-8 -*-
"""
Created on Sat Mach 11 2023

@author: Mauricio Cespedes Tenorio
"""

import numpy as np
import cv2
import os
import random
# import matplotlib.pyplot as plt
import zipfile
import os


imageSize = 128
noise_factor = 0.1

source_path = '/project/ctb-akhanf/myousif9/Neural_Networks_project'
imagePath = os.path.join(source_path, "output/Train/resampled.zip")
outZip = os.path.join(source_path, "output/Train/deblurring.zip")

# New zip file
zipf = zipfile.ZipFile(outZip, 'w', zipfile.ZIP_DEFLATED)

with zipfile.ZipFile(imagePath, mode="r") as archive:
    n_files = len(archive.namelist())
    x_train = np.zeros((n_files,imageSize,imageSize,3))
    for filename in archive.namelist():
        data = archive.read(filename)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        image = cv2.GaussianBlur(image, (0,0),2)
        retval, buf = cv2.imencode('.jpeg', image)
        zipf.writestr(filename, buf)
zipf.close()
print("All images were read!!!")  
