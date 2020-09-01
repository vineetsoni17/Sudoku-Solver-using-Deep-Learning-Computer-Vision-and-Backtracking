import cv2
import numpy as np 
import os 

y=0
for i in os.listdir():
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]
    width = img.shape[1]
    for j in range(0,9):
        for k in range(0,9):
            x = img[height*j-1:height*(j+1)-1, width*k-1:width*(k+1):]
    y+=1