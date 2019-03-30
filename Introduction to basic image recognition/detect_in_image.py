import cv2 # Import OpenCv

img  = cv2.imread("images/mainpic.jpg")# Read the image file

img = cv2.resize(img, (800, 600))# Resizing picture 840 px X 600 px

grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# Convert image into grayscale

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")# Initialize haar cascade feature for frontalface

faces = face_classifier.detectMultiScale(grayimg, scaleFactor=1.3, minNeighbors=3)# Detect faces on image

# Draw the rectangle around the face
for (x,y, w, h) in faces:#loop through the faces and draw rectangle and circles
     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the output image with face detected

cv2.imshow("Face detected", img)#display the image
cv2.waitKey(0)#0 = wait for key stroke indefinitely

"""
Details about detectMultiScale(src, scaleFactor, minNeighbors) fuction

Image: The first input is the grayscale image.

scaleFactor: This scale factor is used to create scale pyramid as shown in the picture. 
            Suppose, the scale factor is 1.03, it means we're using a small step for resizing, 
            i.e. reduce size by 3 %, we increase the chance of a matching size with the model for detection is found
            
minNeighbors: Detection algorithm that uses a moving window to detect objects, it does so by defining 
                how many objects are found near the current one before it can declare the face found.
                Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
                This parameter will affect the quality of the detected faces: higher value results in less detections 
                but with higher quality. We're using 5 in the code.
"""