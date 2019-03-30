# Import OpenCv
# Initialize webcam
# Open a template image and get its width and height
# Create a loop to continuously read the video
# 	Resize or flip the video if required
# 	Convert the video into grayscale and also blur it
# 	Match image template with the blurred video
# 	Find the area where the template is matched for maximum time
# 	Draw the rectangle around the area in the video, where the template is matching
# 	Display the video with the rectangle
# 	Release the webcam when you close the program

import cv2
# 0 = embedded webcam, 1 = external camera
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)# creating variable to capture video
templateImg = cv2.imread("images/arrowTmpl.jpg", 0)#Initialize haar cascade feature for frontalface
imgW, imgH = templateImg.shape[::-1]

while (True):
    read, frame = webcam.read()# capturing video from webcam
    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# Converting video to Grayscale
    blurVideo = cv2.GaussianBlur(grayVideo, (11,11), 0)# Applying gaussian blur

    result = cv2.matchTemplate(blurVideo,templateImg, cv2.TM_CCOEFF_NORMED)# Comparing video with template
    # Storing coordinates of template matching area
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(result)# minMaxLoc correlation between the pixel in template and the test source
    topLeft = maxLocation

    if maxValue>0.5:
        bottom_right = (topLeft[0]+imgW, topLeft[1]+imgH)
        # Drawing rectangle on the template matched pixel area
        cv2.rectangle(frame, topLeft, bottom_right, (0, 255, 0), 3)

    cv2.imshow("Video", frame)  # Displaying Final video
    if cv2.waitKey(1) == ord("q"):  # Defining letter "q" to close program when pressed
        break

webcam.release()
cv2.destroyAllWindows()






























