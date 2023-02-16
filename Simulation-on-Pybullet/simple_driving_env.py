import gym
from gym import spaces
import numpy as np
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import time
import pybullet as p
from car import Car
from plane import Plane
from line_track import track

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode = 'direct',track = 0):
        
        self.car = None
        self.goal = None
        self.done = False
        self.terminated = False
        self.rendered_img = None
        self.render_rot_matrix = None
        self.frame = 0
        self.w = 600
        self.h = 600
        self.timeSet = 1/30
        self.action_space = spaces.Discrete(3)
        # self.observation_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(4)
        self.np_random, _ = gym.utils.seeding.np_random()
        self.previous_goodness = 0
        self.reward_window = (np.arange(20,32), 
                              np.arange(5,27)) 
        self.reward = 0
        self.track = track 
        self._action_to_direction = {
            0: np.array([0.1, 0]),   #stright
            1: np.array([0.1, -6]),  #left
            2: np.array([0.1,  6]),  #right
            3: np.array([0, 0]),     #stop 
        }
 
        if mode == 'direct':
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)
    
        p.setTimeStep(self.timeSet, self.client)
        self.reset()
  

    def getState(self,obs):
        # print(obs)
        if obs <= 12: state = 0                     # left
        elif obs >= 12  and obs <=20: state =  1    # stright
        elif obs >= 20 : state = 2                  # right
        return state
    
    def map(self,x,  in_min,  in_max,  out_min,  out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    

    def RewardFunction(self,binary):
        goodness = 0  # amount of white pixels in processed image
        reward = 0    # current reward
        for r in self.reward_window[0]:
            for c in self.reward_window[1]:
                goodness += (binary[r][c] == 1.0)
        
        if goodness > self.previous_goodness: 
            state = 0
            reward = 2
        elif (goodness < self.previous_goodness) or (goodness == self.previous_goodness and goodness == 0): 
            state = 1
            reward = -1
            
        elif (goodness >= 65 and goodness <=70 ) :
            state = 2
            reward = 5 
            done = True
        else:
            state = 3
            reward = 0 
               
        self.previous_goodness = goodness
        return (state,reward,done)
    

    def step(self, action):
        direction = self._action_to_direction[action]
        self.car.apply_action(direction)
        p.stepSimulation()
        car_ob = self.car.get_observation()
        state,self.reward,done = self.RewardFunction(car_ob)
        info = {}
        ob = np.array(state,dtype=int)
        return ob, self.reward, done, info

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        Plane(self.client)
        
        if self.track == 0:
            track(self.client,0)
            self.car = (Car(self.client))
        else :
            track(self.client,1)
            self.car = (Car(self.client,-1.6))
        
        self.done = False
        self.terminated = False 

        car_ob = self.car.get_observation()
        state,self.reward,self.done = self.rewardFunction(car_ob)
        #reward,done,terminated = self.rewardFunction(state,noLineDetection)
        return (np.array(state, dtype=int),self.done)

    def render(self, mode='direct'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((self.w, self.h, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                    nearVal=0.01, farVal=100)
        position, orientation = [list(l) for l in
        p.getBasePositionAndOrientation(car_id, client_id)]
            
        x, y, z = position
        orientationv1 = p.getEulerFromQuaternion(orientation)
        pitch, roll, yaw = orientationv1
            
        cameraPos = [0]*3
        distance =  0.4 
        cameraPos[0] = x + math.cos(yaw) * distance
        cameraPos[1] = y + math.sin(yaw) * distance
        cameraPos[2] = z + 0.2
            
        rot_mat = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, -2])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            
        view_matrix = p.computeViewMatrix(cameraEyePosition = cameraPos, 
                                            cameraTargetPosition = cameraPos + camera_vec, 
                                            cameraUpVector  = up_vec)

        self.frame = p.getCameraImage(self.w, self.h,
                                          view_matrix,
                                          proj_matrix)[2]
            
        frame = np.reshape(self.frame, (self.w, self.h, 4))
            
        cv2.imshow("display",frame)
        cv2.waitKey(1)
        plt.draw()
        plt.pause(.0001)

    def close(self):
        p.disconnect(self.client)
