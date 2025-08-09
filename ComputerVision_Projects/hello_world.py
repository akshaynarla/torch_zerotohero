import cv2 as cv
import sys

# This script reads an image and displays it in a window.
img = cv.imread('Images/StaryNight.jpg')
print("Size of loaded image:", img.shape)

img = cv.resize(img, (800, 600))   # resize the image to 800x600 pixels. 800 is the width and 600 is the height.
print("Size of resized image:", img.shape)

img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) # convert the image to grayscale
print("Size of converted image:", img.shape)   #  Shape is (height, width, channels) where channels is 1 for grayscale 

if img is None:
    sys.exit('Could not read the image.')
# shows the read img in a window
cv.imshow('Display window', img)

k = cv.waitKey(0) & 0xFF
# upon pressing s, the image is saved as starry_night.png
# if any other key is pressed, the window closes without saving
if k == ord('s'):
    cv.imwrite('Images/starry_night.png', img)

