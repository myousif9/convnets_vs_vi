# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:49:35 2018

@author: root
"""

import numpy as np
import cv2
import zipfile
import os

source_path = '/project/ctb-akhanf/myousif9/Neural_Networks_project'
outZip = os.path.join(source_path, "output/Train/resampled.zip")
imagePath = os.path.join(source_path, 'sourcedata/CLS-LOC_val_dataset.zip')
# New zip file
zipf = zipfile.ZipFile(outZip, 'w', zipfile.ZIP_DEFLATED)

with zipfile.ZipFile(imagePath, mode="r") as archive:
    n_files = len(archive.namelist())
    illusions2 = np.zeros((n_files,128,128,3))
    for filename in archive.namelist():
        data = archive.read(filename)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        image = cv2.resize(image,(128,128))
        retval, buf = cv2.imencode('.jpeg', image)
        zipf.writestr(filename, buf)
        # illusions2[i-1,:,:,:] = image
zipf.close()
# for i in range(1,22):
    
#     image = cv2.resize(cv2.imread(imagePath+str(i)+'.jpg'),(128,128))
#     illusions2[i-1,:,:,:] = image
    
#     cv2.imwrite(savePath+str(i)+'.jpg',image)

# np.save(savePath+'illusionsSup',illusions2)