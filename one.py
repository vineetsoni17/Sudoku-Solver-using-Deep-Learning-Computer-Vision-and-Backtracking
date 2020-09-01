import cv2
import numpy as np
import matplotlib.pyplot as plt

def fun(): 
    
    
    cap=cv2.VideoCapture(0)
    while True:
        cv2.createTrackbar('bw_threshold','image', 0, 255, fun)
        cv2.createTrackbar('u_t','image', 0, 1000, fun)
        cv2.createTrackbar('l_t', 'image', 0, 100, fun)
        a=cv2.getTrackbarPos('bw_threshold', 'image')
        b=cv2.getTrackbarPos('u_t', 'image')
        c=cv2.getTrackbarPos('l_t', 'image')
        f, im= cap.read()
        bw=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        x=cv2.GaussianBlur(bw,(3,3),cv2.BORDER_DEFAULT)
        ret, inv=cv2.threshold(x, a, 255, cv2.THRESH_BINARY_INV)
        canny=cv2.Canny(inv, b, c)
        contour, hierarchy=cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, contour, -1, (0, 0, 255), 4)
        cv2.imshow('Feed', im)
        cv2.imshow('Feed1', canny)
        cv2.imshow('Feed2', inv)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()

    # bw=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # x=cv2.GaussianBlur(bw,(1,1),cv2.BORDER_DEFAULT)
    # ret, inv=cv2.threshold(x, 170, 255, cv2.THRESH_BINARY_INV)
    # canny=cv2.Canny(inv, 50, 50)
    lines=(cv2.HoughLines(canny, 1, np.pi/180, 200))

    grid_marked=im
    for i in range(0,len(lines)):
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
    plt.imshow(grid_marked)
    plt.show()

fun()