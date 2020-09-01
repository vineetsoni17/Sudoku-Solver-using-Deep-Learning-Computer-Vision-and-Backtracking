import cv2
import numpy as np
import matplotlib.pyplot as plt

u="WhatsApp Image 2020-08-28 at 1.22.23 PM.jpeg"
im=cv2.imread(u)
print(type(im))
bw=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
x=cv2.GaussianBlur(bw,(1,1),cv2.BORDER_DEFAULT)
ret, inv=cv2.threshold(x, 170, 255, cv2.THRESH_BINARY_INV)
canny=cv2.Canny(inv, 200, 150)
lines=cv2.HoughLines(canny, 1, np.pi/180, 300)

grid_marked=im
for i in range(len(lines)):
    for dist, theta in lines[i]:
        a=np.cos(theta)
        b=np.sin(theta)
        x=a*dist
        y=b*dist
        x1=int(x+(2000*(-b)))
        y1=int(y+(2000*(a)))
        x2=int(x-(2000*(-b)))
        y2=int(y-(2000*(a)))
        cv2.line(grid_marked, (x1,y1), (x2,y2), (0,255,0), 3)

cv2.imwrite('grid_marked.png', grid_marked)
plt.imshow(canny, cmap='gray')
plt.show()



