import cv2 as cv
import sys

# This script reads an image and displays it in a window.
img = cv.imread('Images/people1.jpg')
img = cv.resize(img, (720, 480))  # Resize the image to 640x48
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# print("Size of loaded image:", img.shape)

if img is None:
    print("Could not read the image.")
    sys.exit()

# Load the Haar Cascade for face detection. This is an already pre-built adaboost classifier.
# mathematically trained to detect faces in images.
face_detector = cv.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(img)
print("Number of faces detected:", len(detections), "Face coordinates:", detections)  
# detections is  [x , y, width, height] for each face detected in the image coordinates

# Draw rectangles around detected faces
for (x, y, w, h) in detections:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)   # BGR (255,0,0) is set and you get blue rectangle on the detected faces. 2 is the thickness.

cv.imshow('Display window', img)
k = cv.waitKey(0) & 0xFF