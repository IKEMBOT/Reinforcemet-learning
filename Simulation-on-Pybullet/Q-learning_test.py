import time
import pybullet as p
import gym
import numpy as np
import time
import math
from simple_driving_env import SimpleDrivingEnv
import matplotlib.pyplot as plt
import codecs, json

file_path =  "Qvalue.json"
input = codecs.open(file_path, 'r', encoding='utf-8').read()
load_Q = json.loads(input)
Q = np.array(load_Q[0]['Q_value'])
print(Q)              
env = SimpleDrivingEnv("human",track=1)
s,_  = env.reset()
while True:
    a = np.argmax(Q[s,:]).item()
    print(a)
    print(f"Chose action {a} for state {s}") 
    s, reward, terminated , done = env.step(a)
    env.render
    if terminated or done:
        print("Finished!", reward)
        time.sleep(5)
        break
env.close()

