import pyscreenshot # for capturing the screen
import pytesseract  # for the OCR
import cv2
import numpy as np
import json

class Debugger:
    def __init__(self, debug):
        self.debug = debug

    def start_debug(self):
        if self.debug:
            # debug screen grabbing
            # grab a screenshot of the game
            screen = pyscreenshot.grab(bbox=(630, 360, 1280, 780))  # cords for 1920x1080 screen
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)
            print(f"Screenshot shape: {screen.shape}")
            cv2.imshow("Game", screen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            config = json.load(open("config.json"))

            # find the score; look for game over
            # screenshot size is 420x600
            # cords to take the score
            reward_cords = config['cords']['reward']
            x1 = reward_cords[0]
            x2 = reward_cords[1]
            y1 = reward_cords[2]
            y2 = reward_cords[3]
            cv2.imshow("Game", screen[y1:y2, x1:x2])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            score = pytesseract.image_to_string(screen[y1:y2, x1:x2])
            print(score)

            # observation
            observation_cords = config['cords']['observation']
            x1 = observation_cords[0]
            x2 = observation_cords[1]
            y1 = observation_cords[2]
            y2 = observation_cords[3]
            screen = screen[y1:y2, x1:x2]
            # screen = cv2.resize(screen, (32, 50))
            cv2.imshow("Game", screen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # test if lost
            screen = pyscreenshot.grab(bbox=(630, 360, 1280, 780))  # cords for 1920x1080 screen
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)
            lost_cords = config['cords']['lost']
            x1 = lost_cords[0]
            x2 = lost_cords[1]
            y1 = lost_cords[2]
            y2 = lost_cords[3]
            screen = screen[y1:y2, x1:x2]
            text = pytesseract.image_to_string(screen)
            cv2.imshow("Game", screen)
            cv2.waitKey(0)
            print(text)