from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import matplotlib.pyplot as plt
import keyboard

frame_save_path = "frames" 
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
earThresh = 0.3  # distance between vertical eye coordinate Threshold
earFrames = 30 # consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# get the coord of left & right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

total_frames = 0
drowsy_frames = 0

# Initialize a flag to control the loop
running = True

while running:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        if ear < earThresh:
            count += 1

            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                drowsy_frames += 1
        else:
            count = 0

    total_frames += 1
    accuracy = (1 - drowsy_frames / total_frames) * 100

    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    frame_filename = f"frame_{total_frames:04d}.jpg"
    cv2.imwrite(os.path.join(frame_save_path, frame_filename), frame)

    # Display the frame using OpenCV's imshow function
    cv2.imshow("Frame", frame)

    # Check if user wants to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
