import numpy as np
import cv2

square_pts = []
def get_coordinates(event, x, y, flags, param):
    global square_pts, count
    if event == cv2.EVENT_LBUTTONDOWN:
        square_pts.append((x, y))
        count = count+1
        cv2.circle(thresh_inv, square_pts[-1], radius = 5, color = (0, 0, 0), thickness = -1)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    thresh_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    cv2.imshow('Video Feed', thresh_inv)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

count = 0
cv2.namedWindow('Final Image')
cv2.setMouseCallback('Final Image', get_coordinates)

while count < 4:
    cv2.imshow('Final Image', thresh_inv)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

square_pts = np.float32(square_pts)
dst = np.float32([[0,0],[0,252],[252,252],[252,0]])

M = cv2.getPerspectiveTransform(square_pts, dst)
transform = cv2.warpPerspective(thresh_inv, M, dsize=(252,252))

cv2.namedWindow('Transformed Image')
cv2.imshow('Transformed Image', transform)
cv2.waitKey(0)