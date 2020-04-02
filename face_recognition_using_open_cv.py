# Face Recognition
import numpy as np
import cv2
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
    return frame # We return the image with the detector rectangles.

model = "resnet-34_kinetics.onnx"
CLASSES=open("action_recognition_kinetics.txt").read().strip().split("\n")
# input_video = "example_activities.mp4"

SAMPLE_DURATION = 16  # duration (i.e., # of frames for classification)
SAMPLE_SIZE = 112

net = cv2.dnn.readNet(model)

input_video = "example.mp4"
video_capture = cv2.VideoCapture(input_video) # We turn the webcam on.

while True: # We repeat infinitely (until break):
    frames = []
    _, frame = video_capture.read() # We get the last frame.
    frame = imutils.resize(frame, width=400)
    frames.append(frame)

    # now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0,(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]

    print(label)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.

    cv2.imshow('Video', canvas) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
