import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2 
import torch

import one
import three 
import four

img = cv2.imread("puzzle.jpg")
q = 0
grid = np.zeros((9, 9))
height = img.shape[0]//9
width = img.shape[1]//9
for j in range(0,9):
    for k in range(0,9):
        x = img[height*j+1:height*(j+1)+1, width*k+1:width*(k+1)+1]
        a = cv2.resize(x, (50, 50), interpolation=cv2.INTER_AREA)
        index = three.real_deal(a)
        grid[j, k] = index
        q +=1

print(grid)

# ret, inv=cv2.threshold(x, 170, 255, cv2.THRESH_BINARY_INV)
# canny=cv2.Canny(inv, 200, 150)
# lines=cv2.HoughLines(canny, 1, np.pi/180, 300)

# grid_marked=im
# for i in range(len(lines)):
#     for dist, theta in lines[i]:
#         a=np.cos(theta)
#         b=np.sin(theta)
#         x=a*dist
#         y=b*dist
#         x1=int(x+(2000*(-b)))
#         y1=int(y+(2000*(a)))
#         x2=int(x-(2000*(-b)))
#         y2=int(y-(2000*(a)))
#         cv2.line(grid_marked, (x1,y1), (x2,y2), (0,255,0), 3)

# cv2.imwrite('grid_marked.png', grid_marked)
# plt.imshow(canny, cmap='gray')
# plt.show()



