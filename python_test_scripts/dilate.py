import numpy as np
import cv2

img = cv2.imread('mgn.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

#===================================================================
# Dilate to find the search boundary

kernel1 = np.ones((15, 15), np.uint8)
dilated1 = cv2.dilate(gray, kernel1, iterations=1)

kernel2 = np.ones((13, 13), np.uint8)
dilated2 = cv2.dilate(gray, kernel2, iterations=1)

delta_dilated = dilated1-dilated2

# sobelx64f = cv2.Sobel(gray, cv2.CV_64F,1,1,ksize=3)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# cv2.imshow('sobel', sobel_8u)


# laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# laplacian = np.uint8(np.absolute(laplacian))
# laplacian[gray!=255] =0
# cv2.imshow('tip_route', laplacian)

# We use blurring convolution filter, which invole 3x3 convolution matrix, 
# instead of normal gradient detection, due to gradient,
# detection does NOT take account in 4 corner values

kernel_conv = np.ones((5, 5), np.float32)/6.9
convoluted = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel_conv)
convoluted[gray!=255] = 0
cv2.imshow('convolute', convoluted)
tip_end_removed = np.zeros_like(convoluted)
tip_end_removed[np.bitwise_and(convoluted<150, convoluted>10)] = 255

tip_end_dilated = cv2.dilate(tip_end_removed, kernel2, iterations=2)

delta_dilated = dilated1-dilated2-tip_end_dilated
cv2.imshow('dilated', delta_dilated)

#===================================================================

#===================================================================
# Search boundary contour detetion
ret, thresh = cv2.threshold(delta_dilated, 150, 255, cv2.THRESH_BINARY)
# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
boundary_contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# draw contours on the original image
img_bound_contour = delta_dilated.copy()
img_bound_contour = cv2.cvtColor(img_bound_contour, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image=img_bound_contour, contours=boundary_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
# see the results
cv2.imshow('img_contour', img_bound_contour)

# Find original path contours
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
path_contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# draw contours on the original image
img_path_contour = delta_dilated.copy()
img_path_contour = cv2.cvtColor(img_path_contour, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image=img_path_contour, contours=path_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
# see the results
cv2.imshow('img_contour', img_path_contour)

#===================================================================

#===================================================================
# Mask for visualization

output = np.copy(img)
r1, g1, b1 = 255, 255, 255 # Original value
r2, g2, b2 = 255, 255, 0 # Value that we want to replace it with

red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]
mask = (red == r1) & (green == g1) & (blue == b1)

output[:,:,:3][mask] = [r2, g2, b2]

output = output + cv2.cvtColor(delta_dilated, cv2.COLOR_GRAY2BGR)
cv2.imshow('output', output)

#===================================================================

#===================================================================
# Miscellaneous

cv2.waitKey(0)
cv2.destroyAllWindows()