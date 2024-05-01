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
env = GameEnv1()
total_training_time = 60*60*8  # time in seconds
timestep_coef = 2
done = False

model_save = "model"
model_load = ""

# determine if we are learning or playing
if config['learning'] == 'True':
    """Start learning process"""
    start_time = time.time()
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)

    if model_load != "":
        model.load(model_load)
        print(f"Model {model_load} loaded")

    remaining_time = total_training_time
    remaining_timesteps = int(total_training_time * timestep_coef)

    print("Learning commences...")

    while not done:
        try:
            model.learn(total_timesteps=remaining_timesteps)
            done = True
            break
        except KeyboardInterrupt:
            print("Model saved")
            elapsed_time = time.time() - start_time
            if model_load != "":
                model.save(model_load + "+" + str(round(elapsed_time / 3600, 2)))
            else:
                model.save(model_save + str(round(elapsed_time / 3600, 2)))
            done = True
            break
        except Exception as e:
            print(f"Error: {e}")
            model.save("temp")

            remaining_time = total_training_time - (time.time() - start_time)
            remaining_timesteps = int(remaining_time * timestep_coef)

    elapsed_time = time.time() - start_time
    print(f"Learning session completed. Time elapsed: {elapsed_time} seconds")
    if model_load != "":
        model.save(model_load + "+" + str(round(elapsed_time / 3600, 2)))
    else:
        model.save(model_save + str(round(elapsed_time / 3600, 2)))
else:
    """Start playing with model"""
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    model.load(model_load)
    print(f"Model {model_load} loaded")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
