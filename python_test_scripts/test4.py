import numpy as np
import cv2

img = cv2.imread('mgn.png')
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#===================================================================
def viz_with_original(original, new_line):
    # Mask for visualization
    if len(original) == 2:
        original= cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    output = np.copy(original)
    r1, g1, b1 = 255, 255, 255 # Original value
    r2, g2, b2 = 255, 255, 0 # Value that we want to replace it with

    red, green, blue = original[:,:,0], original[:,:,1], original[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)

    output[:,:,:3][mask] = [r2, g2, b2]

    output = output + cv2.cvtColor(new_line, cv2.COLOR_GRAY2BGR)
    cv2.imshow('output', output)

#===================================================================
# Find tip ends of a line. A line must have TWO ends:

def find_tip_end(line_img):
    kernel_conv = np.ones((5, 5), np.float32)/6.9
    convoluted = cv2.filter2D(src=line_img, ddepth=-1, kernel=kernel_conv)
    convoluted[gray!=255] = 0

    tip_end_removed = np.zeros_like(convoluted)
    tip_end_removed[np.bitwise_and(convoluted<150, convoluted>10)] = 255

#===================================================================
# Dilate to find the search boundary

def find_boundary(path, width=25):
    kernel1 = np.zeros((23,23), np.uint8)
    cv2.circle(kernel1,(11,11), 10, 255, -1)
    dilated1 = cv2.dilate(path, kernel1, iterations=1)

    kernel2 = np.zeros((23,23), np.uint8)
    cv2.circle(kernel2,(11,11), 9, 255, -1)
    dilated2 = cv2.dilate(path, kernel2, iterations=1)

    boundaries = dilated1-dilated2

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

    tip_end_removed = np.zeros_like(gray)
    tip_end_removed[np.bitwise_and(convoluted<150, convoluted>10)] = 255

    kernel3 = np.zeros((21,21), np.uint8)
    cv2.circle(kernel3,(10,10), 11, 255, -1)
    tip_end_removed_dilated = cv2.dilate(tip_end_removed, kernel3, iterations=1)

    boundaries[tip_end_removed_dilated==255] = 0

    return boundaries

#===================================================================
# Find contours

def find_contours(boundaries):
    ret, thresh = cv2.threshold(boundaries, 150, 255, cv2.THRESH_BINARY)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    boundary_contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return boundary_contours

#===================================================================
# Main
boundaries = find_boundary(gray)
viz_with_original(img, boundaries)
boundary_contours = find_contours(boundaries)
#------------------- Visualize boundary contours -------------------
img_bound_contours = img.copy()
cv2.drawContours(image=img_bound_contours, contours=boundary_contours, \
                 contourIdx=-1, color=(0, 255, 0), thickness=2, \
                 lineType=cv2.LINE_AA)
cv2.imshow('contours', img_bound_contours)
#-------------------------------------------------------------------

INIT_POS = np.array([99,330])
INIT_ROT = -0.5 #Radian

cv2.waitKey(0)
cv2.destroyAllWindows()