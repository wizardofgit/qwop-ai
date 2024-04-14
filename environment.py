"""Contains the environment class for the game."""
import cv2
import numpy as np
import mss
import pytesseract  # for the OCR
import pyautogui  # for the mouse and keyboard control
import gym
import json


class GameEnv1(gym.Env):
    """Custom environment for the game.
    It uses Discrete action space (can use only one key per action)."""

    def __init__(self, debug=False):
        # [1,4] - push keys down, [5,8] - release keys
        self.action_map = {
            0: '',
            1: 'q',
            2: 'w',
            3: 'o',
            4: 'p',
            5: 'q',
            6: 'w',
            7: 'o',
            8: 'p',

        }
        # define the action space (possible model actions) as discrete, choosing integer values from 0 to 4
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(50, 32), dtype=np.uint8)
        self.current_observation = None
        self.current_screen = None
        self.ticks_without_progress = 0  # used to determine if it should reset
        self.previous_score = 0
        self.sct = mss.mss()
        self.monitor = {"top": 360, "left": 630, "width": 650, "height": 420}
        self.debug = debug
        self.config = json.load(open("config.json"))
        self.no_progress_coef = 2 # how much to penalize for no progress

    def get_screen(self):
        screen = self.sct.grab(self.monitor)
        screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)

        return screen

    def truncated(self):
        """Returns if the game is truncated."""
        if self.ticks_without_progress > 20:
            if self.debug:
                print("Too long without progress, resetting the game")
            return True
        return False

    def lost(self, screen=None):
        if screen is None:
            screen = self.sct.grab(self.monitor)
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)

        lost_cords = self.config['cords']['lost']
        x1 = lost_cords[0]
        x2 = lost_cords[1]
        y1 = lost_cords[2]
        y2 = lost_cords[3]
        screen = screen[y1:y2, x1:x2]
        text = pytesseract.image_to_string(screen)

        if 'participant' in text.lower():
            if self.debug:
                print(f"Game lost: {text}")
            return True
        return False

    def get_observation(self, screen=None):
        """Returns the current observation."""
        if screen is None:
            screen = self.sct.grab(self.monitor)
            screen = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2GRAY)

        observation_cords = self.config['cords']['observation']
        x1 = observation_cords[0]
        x2 = observation_cords[1]
        y1 = observation_cords[2]
        y2 = observation_cords[3]

        screen = screen[y1:y2, x1:x2]
        observation = cv2.resize(screen, (32, 50))

        return observation

    def get_reward(self, observation=None):
        """Returns the reward."""
        if observation is None:
            observation = self.sct.grab(self.monitor)
            observation = cv2.cvtColor(np.array(observation), cv2.COLOR_BGR2GRAY)

        reward_cords = self.config['cords']['reward']
        x1 = reward_cords[0]
        x2 = reward_cords[1]
        y1 = reward_cords[2]
        y2 = reward_cords[3]
        score = pytesseract.image_to_string(observation[y1:y2, x1:x2])
        score = float(score.split(' ')[0])

        if score > self.previous_score:
            self.ticks_without_progress = 0

            if score > 0:
                reward = score
            else:
                reward = score * -1
        else:
            if score > 0:
                reward = score * -1
            else:
                reward = score

        reward -= self.no_progress_coef * self.ticks_without_progress

        if self.debug:
            print(f"Score: {score}, Reward: {reward}")

        return reward

    def step(self, action):
        """Takes action as argument and returns the next observation, reward, done and info."""
        action = int(action)

        if action == 0:
            pass
        elif action < 5:
            pyautogui.keyDown(self.action_map[action])
        else:
            pyautogui.keyUp(self.action_map[action])
        if self.debug:
            print(f'Action: {self.action_map[action]}')

        self.current_screen = self.get_screen()
        self.current_observation = self.get_observation(self.current_screen)
        reward = self.get_reward(self.current_screen)
        info = {}
        terminated = self.lost(self.current_screen)
        truncated = self.truncated()

        if self.previous_score > reward:
            self.ticks_without_progress += 1
        if self.previous_score - reward < 0.2:
            self.ticks_without_progress += 1
        if self.previous_score - reward >= 0.2:
            self.ticks_without_progress = 0

        self.previous_score = reward

        return self.current_observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment and returns the initial observation."""
        pyautogui.press('r')
        if self.debug:
            print('Game restarted')

        self.current_screen = self.get_screen()
        self.current_observation = self.get_observation(self.current_screen)
        self.ticks_without_progress = 0
        self.previous_score = 0

        return self.current_observation, {}

    def render(self, mode='human'):
        """As per specification is mandatory, but not used in this project."""
        pass
