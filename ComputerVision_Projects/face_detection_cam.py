import cv2 as cv
import sys

# not accurate, but shows the capability of just opencv without any machine learning
# This script uses the webcam to detect faces and eyes in real-time.

img = cv.VideoCapture(0)  # Use the webcam for live face detection
if img is None:
    print("Could not read the image.")
    sys.exit()
 
# Load the Haar Cascade for face detection. This is an already pre-built adaboost classifier.
# mathematically trained to detect faces in images.
face_detector = cv.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_detector = cv.CascadeClassifier('Cascades/haarcascade_eye.xml')
count = 0

while True:
    frame = img.read()[1]  # Read a frame from the webcam
    img_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to RGB
    print("Processing frame number:", count)
    detections = face_detector.detectMultiScale(img_frame, minNeighbors=5, minSize=(28, 28))    
    eye_detections = eye_detector.detectMultiScale(img_frame, minNeighbors=3, minSize=(8,8))

    print("Number of faces detected:", len(detections), "Face coordinates:", detections)  
    print("Number of eyes detected:", len(eye_detections), "Eye coordinates:", eye_detections)  

    # Draw rectangles around detected faces
    for (x, y, w, h) in detections:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 

    for (x, y, w, h) in eye_detections:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) 

    cv.imshow('Display window', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):   # with 0 only 1 frame is processed, with 1, it processes continuously
        print("Exiting the loop.")
        break  # Exit the loop if any key is pressed
    count+= 1


img.release()  # Release the webcam
cv.destroyAllWindows()  # Close all OpenCV windows