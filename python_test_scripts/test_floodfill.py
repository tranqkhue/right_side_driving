import cv2
import numpy as np

img = np.full((300,300), 0).astype(np.uint8)
cv2.circle(img, (150,150), 50, 255, 5)
cv2.circle(img, (150,150), 25, 255, 5)

img = cv2.floodFill(img, None, (1,1), 255)[1]

cv2.imshow('test', img.astype(np.uint8))
cv2.waitKey(0)

cv2.destroyAllWindows()