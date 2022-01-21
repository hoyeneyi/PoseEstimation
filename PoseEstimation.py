import numpy as np
import cv2
import mediapipe as mp
import time
from playsound import playsound
import numpy as np
import pyttsx3
import pygame
import time
import math
from numpy.lib import utils

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
drawing = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# initialize the person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

vid = cv2.VideoCapture(0)

# the output written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

# vid = cv2.VideoCapture("PoseVideos/squat.mp4")
prevTime = 0

# new variables
lst = []
n = 0
scale = 3
count = 0
brake = 0
x = 150
y = 195


def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 150)

    engine.setProperty('voice', voices[0].id)
    engine.say(audio)

    engine.runAndWait()


speak("I will now measure your height")

while True:
    success, img = vid.read()
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # returns bounding boxes for detected people
    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the color picture
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    out.write(img.astype('uint8'))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print the body point location as well as how visible they are
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            lst[n] = lst.append([id, lm.x, lm.y])
            n + 1
            h, w, c = img.shape
            print(id, lm)
            if id == 32 or id == 31:
                cx1, cy1 = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx1, cy1), 10, (255, 0, 0), cv2.FILLED)
                d = ((cx2 - cx1) ** 2 + (cx2 - cy1) ** 2) ** 0.5
                di = round(d * 0.5)
                print(di)
                pygame.mixer.init()

                speak(f"You are {di} centimeters tall")
                speak("I am done")
                if ord('q'):
                    cv2.cv2.destroyAllWindows()
                break
                dom = ((lm.z - 0) ** 2 + (lm.y - 0) ** 2) ** 0.5

            if id == 6:
                cx2, cy2 = int(lm.x * w), int(lm.y * h)

                cy2 = cy2 + 20
                cv2.circle(img, (cx2, cy2), 15, (0, 0, 0), cv2.FILLED)

    cv2.putText(img, "Height : ", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), thickness=2)
    cv2.putText(img, str(di), (180, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), thickness=2)
    cv2.putText(img, "cms", (240, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), thickness=2)
    cv2.putText(img, "Stand at least 3 meter away", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                thickness=2)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS : {fps}", (40, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2)
    # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    # vid.release()
    # out.release()
    # cv2.destroyAllWindows()
    cv2.waitKey(1)
