import cv2 as cv
import sys
import numpy as np

# This script reads an image and displays it in a window.
img = cv.imread('Images/gabriel.png')
if img is None:
    print('Could not open or find the image:', sys.argv[1])
    sys.exit()

img = np.float32(img)/255.0  # Convert to float32 and normalize for better processing

# Calculate gradient
gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)  # goes along x-axis (dx=1) to see if there are changes. i.e., vertical edges will be identified
gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)  # goes along y-axis (dy=1) to see if there are changes. i.e., horizontal edges will be identified
mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)  # mag has the magnitude i.e. how strong the edge is (sqrt(gx2 + gy2)), angle has the direction of the edge
# 2 values per pixel: HOG uses the magnitude and angle to create a histogram of oriented gradients. Histogram is a 9-bin, with 8x8 it is calculated for each 8x8 pixel block.
# This is a design decision.
# edge detection can also be done with cv.Canny(img) but it is not used here.

# Concatenate images for display
Hori = np.concatenate((img, gx, gy, mag, angle), axis=1)

cv.imshow('Image', Hori)
k = cv.waitKey(0) & 0xFF
