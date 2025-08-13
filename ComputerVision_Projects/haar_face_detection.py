import cv2 as cv
import sys

# This script reads an image and displays it in a window.
img = cv.imread('Images/people1.jpg')

if img is None:
    print("Could not read the image.")
    sys.exit()
 
img = cv.resize(img, (720, 480))  # because of the resize, the adaboost classifier will not work well.

# Load the Haar Cascade for face detection. This is an already pre-built adaboost classifier.
# mathematically trained to detect faces in images.
face_detector = cv.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_detector = cv.CascadeClassifier('Cascades/haarcascade_eye.xml')
# Same as above, you can use other classifier for different objects. This is not so intelligent and does not learn on its own.

detections = face_detector.detectMultiScale(img, scaleFactor=1.07, minNeighbors=5, minSize=(28, 28))    
# scaleFactor is the parameter that specifies how much the image size is reduced at each image scale.
# smaller values lead to more detections, but also increase the processing time. Particularly useful for small faces.
# with people2.jpg, scaleFactor=1.1 gave 12 out of 14 faces, in the previous settings.
# with grayscale image, scaleFactor=1.1 gave 9 out of 14 faces. 
# with scaleFactor=1.06, better detection of 13 out of 14 faces, but too many false positives.
# minNeighbors is the parameter specifying how many neighbors each candidate rectangle should have to retain it. i.e, more indicates, more possibility of a face based on
# the xml file calculation with cascade classifier
# minSize is the minimum possible object size. Objects smaller than that are ignored.

eye_detections = eye_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(8,8))

print("Number of faces detected:", len(detections), "Face coordinates:", detections)  
print("Number of eyes detected:", len(eye_detections), "Eye coordinates:", eye_detections)  

# Draw rectangles around detected faces
for (x, y, w, h) in detections:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 

for (x, y, w, h) in eye_detections:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1) 

cv.imshow('Display window', img)
k = cv.waitKey(0) & 0xFF