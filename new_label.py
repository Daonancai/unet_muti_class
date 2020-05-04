# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:42:40 2020

@author: xiangzhongyu
"""


import os
import numpy as np
#import cv2
from PIL import Image


path = 'E:\\dataset\\data\\label\\'

imagelist = os.listdir(path)

for i in imagelist:
    imgpath = path + i
    img = Image.open(imgpath)
    img = np.array(img)
    
    

































