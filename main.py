import time
import cv2
import os
import numpy as np
import torch    # for reinforcement learning
import webbrowser
import mss
import json
import pyscreenshot # for capturing the screen
import pytesseract  # for the OCR
import pyautogui    # for the mouse and keyboard control

# make sure CUDA is available
print(f"Is CUDA available? {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

# open config
config = json.load(open("config.json"))

# open the game
webbrowser.open("http://www.foddy.net/Athletics.html")
# wait for the game to load
time.sleep(3)

# click on the game to activate it
pyautogui.click(950, 570)   # cords for 1920x1080 screen

# debug screen grabbing
if config['debug'] == 'True':
    # grab a screenshot of the game
    screen = pyscreenshot.grab(bbox=(630, 360, 1280, 780))  # cords for 1920x1080 screen
    screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    print(f"Screenshot shape: {screen.shape}")
    cv2.imshow("Game", screen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# find the score; look for game over
if config['debug'] == 'True':
    # screenshot size is 420x600
    # cords to take the score
    x1, x2, y1, y2 = 200, 450, 30, 80
    cv2.imshow("Game", screen[y1:y2, x1:x2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    score = pytesseract.image_to_string(screen[y1:y2, x1:x2])
    print(score)

start_time = time.time()
# determine if we are learning or playing
if config['learning'] == 'True':
    """Start learning process"""
    pass
else:
    """Start playing with best yet model"""
    pass