# Import OpenCv
# Initialize webcam
# Initialize haar cascade feature for frontalface
# Create a loop to continuously read the video
# 	Resize or flip the video if required
# 	Convert the video into grayscale and also blur it
# 	Detect faces in the video
# 		Draw the rectangle around the faces in the video
# 	Display the video with the rectangle
# 	Release the webcam when you close the program
import cv2

# 0 = embedded webcam, 1 = external camera
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)# creating variable to capture video
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#Initialize haar cascade feature for frontalface

while (True):
    read, frame = webcam.read()# capturing video from webcam
    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Converting video to Grayscale
    blurVideo = cv2.GaussianBlur(grayVideo, (11, 11), 0) # Applying gaussian blur

    faces = face_classifier.detectMultiScale(blurVideo, 1.3, 3)# Detect faces on webcam video

    # Draw the rectangle around the face
    for (x, y, w, h) in faces:#loop through the faces and draw rectangle
        cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 4)

    # Display the output video with face detected
    cv2.imshow("Video", frame)#display the image
    if cv2.waitKey(1) == ord("q"):  # Defining letter "q" to close program when pressed
        break

webcam.release()
cv2.destroyAllWindows()


























