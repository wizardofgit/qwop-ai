import os.path
import time
import stable_baselines3
import torch    # for reinforcement learning
import webbrowser
import json
import pyautogui    # for the mouse and keyboard control
from environment import GameEnv1
from debug import Debugger

# make sure CUDA is available
print(f"Is CUDA available? {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

# open config
config = json.load(open("config.json"))

# open the game
webbrowser.open("http://www.foddy.net/Athletics.html")
# wait for the game to load
time.sleep(6)

# click on the game to activate it
pyautogui.click(950, 570)   # cords for 1920x1080 screen

if config['debug'] == 'True':
    Debugger(True).debug()

# create the environment
env = GameEnv1()
total_training_time = 8*60*60  # 8 hours
done = False

# determine if we are learning or playing
if config['learning'] == 'True':
    """Start learning process"""
    start_time = time.time()
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    # remove the existing model to start new training session
    if os.path.exists("model.zip"):
        os.remove("model.zip")
    remaining_time = total_training_time
    remaining_timesteps = int(total_training_time*8)  # ~ 8 timestep per 1 second

    while not done:
        try:
            # override model if it exists (continue training existing model after error)
            if os.path.exists("model.zip"):
                model = stable_baselines3.PPO.load("model")
            model.learn(total_timesteps=remaining_timesteps)  # 1000 ~ 2 minutes
            done = True
        except:
            model.save("model")
            remaining_time = total_training_time - (time.time() - start_time)
            remaining_timesteps = int(remaining_time*8)

    model.save("model")
else:
    """Start playing with best yet model"""
    model = stable_baselines3.PPO.load("model")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
