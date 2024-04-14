import os.path
import time
import stable_baselines3
import torch    # for reinforcement learning
import webbrowser
import json
import pyautogui    # for the mouse and keyboard control
from environment import GameEnv1
from debug import Debugger

# open config
config = json.load(open("config.json"))

# make sure CUDA is available
if config['debug'] == 'True':
    print(f"Is CUDA available? {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

# open the game
webbrowser.open("http://www.foddy.net/Athletics.html")
# wait for the game to load
time.sleep(6)

# click on the game to activate it
pyautogui.click(950, 570)   # cords for 1920x1080 screen

if config['debug'] == 'True':
    Debugger(True).start_debug()
    exit()

# create the environment
env = GameEnv1(True)
total_training_time = 8*60*60  # 8 hours in seconds
timestep_coef = 7
done = False

model_save = "temp"
model_load = ""

# determine if we are learning or playing
if config['learning'] == 'True':
    """Start learning process"""
    start_time = time.time()
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)

    if model_load != "":
        model = stable_baselines3.PPO.load(model_load)
        print(f"Model {model_load} loaded")

    remaining_time = total_training_time
    remaining_timesteps = int(total_training_time * timestep_coef)  # ~ 8 timestep per 1 second

    while not done:
        try:
            model.learn(total_timesteps=remaining_timesteps)  # 1000 ~ 2 minutes
            done = True
            model.save(model_save)
            break
        except Exception as e:
            print(f"Error: {e}")
            model.save(model_save)

            remaining_time = total_training_time - (time.time() - start_time)
            remaining_timesteps = int(remaining_time * timestep_coef)

    print(f"Learning session completed. Time elapsed: {time.time() - start_time} seconds")
    model.save(model_save)
else:
    """Start playing with model"""
    model = stable_baselines3.PPO.load(model_load+".zip")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
