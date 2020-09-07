import cv2
import numpy as np 
import os 

y = 0
q = 1135
for i in os.listdir("Database"):
    print(i)
    if i != "sudoku-blankgrid.png":
        continue
    img = cv2.imread(os.path.join('Database', i), cv2.IMREAD_GRAYSCALE)
    # ret, inv = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('img', img)
    # cv2.imshow('inv', inv)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    height = img.shape[0]//9
    width = img.shape[1]//9
    for j in range(0,9):
        for k in range(0,9):
            x = img[height*j+1:height*(j+1)+1, width*k+1:width*(k+1)+1]
            a = cv2.resize(x, (50, 50), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"Database/Dataset/img ({q}).jpg", a)
            q +=1
    y+=1