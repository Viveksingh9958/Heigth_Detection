# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:45:52 2024

@author: FirstESCO
"""

import cv2 as cv
import mediapipe as mp
from playsound import playsound
import numpy as np
import pyttsx3
import pygame
import time

# Initialize Mediapipe and Pyttsx3
mpPose = mp.solutions.pose
mpFaceMesh = mp.solutions.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()
capture = cv.VideoCapture(0)

def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 150)
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()

speak("I am about to measure your height now sir")
speak("Although I reach a precision up to ninety-eight percent")

# Variables
scale = 3
ptime = 0

while True:
    isTrue, img = capture.read()
    if not isTrue:
        break
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        landmarks = result.pose_landmarks.landmark
        h, w, _ = img.shape
        
        # Initialize points for distance calculation
        cx1, cy1 = None, None
        cx2, cy2 = None, None
        
        for id, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            
            if id == 31:
                cx1, cy1 = x, y
                cv.circle(img, (cx1, cy1), 15, (0, 0, 255), cv.FILLED)
            elif id == 32:
                cx2, cy2 = x, y
                cv.circle(img, (cx2, cy2), 15, (0, 255, 0), cv.FILLED)
        
        if cx1 is not None and cx2 is not None:
            # Calculate distance in pixels and approximate height in cm
            d = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
            height_cm = round(d*3)  # Adjust the multiplier as needed
            
            #pygame.mixer.init()
            #pygame.mixer.music.load("check.mp3")
            #pygame.mixer.music.play()
            #speak(f"You are {height_cm} centimeters tall")
            #speak("I am done")
            #speak("You can relax now")
            #speak("Press q and give me some rest now.")
            
            # Display height on the screen
            cv.putText(img, f"Height: {height_cm} cms", (40, 200), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), thickness=2)
            cv.putText(img, "Stand at least 1.5 meters away", (100, 300), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        
    # Display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0
    ptime = ctime
    cv.putText(img, f"FPS: {int(fps)}", (40, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=2)
    
    cv.imshow("Task", img)
    
    # Break the loop when 'q' is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()