import pyscreenshot # for capturing the screen
import pytesseract  # for the OCR
import cv2
import numpy as np

class Debugger:
    def __init__(self, debug):
        self.debug = debug

    def debug(self):
        if self.debug:
            # debug screen grabbing
            # grab a screenshot of the game
            screen = pyscreenshot.grab(bbox=(630, 360, 1280, 780))  # cords for 1920x1080 screen
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)
            print(f"Screenshot shape: {screen.shape}")
            cv2.imshow("Game", screen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # find the score; look for game over
            # screenshot size is 420x600
            # cords to take the score
            x1, x2, y1, y2 = 200, 450, 30, 80
            cv2.imshow("Game", screen[y1:y2, x1:x2])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            score = pytesseract.image_to_string(screen[y1:y2, x1:x2])
            print(score)

            x1, x2, y1, y2 = 100, 450, 80, 410
            screen = screen[y1:y2, x1:x2]
            screen = cv2.resize(screen, (32, 50))
            cv2.imshow("Game", screen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # test if lost
            screen = pyscreenshot.grab(bbox=(630, 360, 1280, 780))  # cords for 1920x1080 screen
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)
            x1, x2, y1, y2 = 340, 410, 260, 310
            screen = screen[y1:y2, x1:x2]
            text = pytesseract.image_to_string(screen)
            cv2.imshow("Game", screen)
            cv2.waitKey(0)
            print(text)