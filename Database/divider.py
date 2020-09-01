import cv2
import numpy as np 
import os 

y=0
for i in os.listdir("Database"):
    if i == "divider.py":
        continue
    img = cv2.imread(os.path.join('Database', i), cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]//9
    width = img.shape[1]//9
    for j in range(0,9):
        for k in range(0,9):
            x = img[height*j+1:height*(j+1)+1, width*k+1:width*(k+1)+1]
            a = cv2.resize(x, (50, 50), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join('Database', f"{str(y)}{str(j)}{str(k)}.jpg"), a)    
    y+=1