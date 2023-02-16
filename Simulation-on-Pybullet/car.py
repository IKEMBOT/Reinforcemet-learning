import pybullet as p
import os
import math
import numpy as np
import cv2


class Car:
    def __init__(self, client,y_world = -1.5):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplecar.urdf')
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[0, y_world, 0.1], 
                              physicsClientId=client)

        self.steering_joints = [0, 2]
        self.drive_joints = [1, 3, 4, 5]
        self.joint_speed = 0
        self.c_rolling = 0.2
        self.c_drag = 0.01
        self.c_throttle = 20
        self.H_low = 87
        self.H_high = 179
        self.S_low= 255
        self.S_high = 255
        self.V_low= 0
        self.V_high = 255
        self.timeSet = 1/30

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        throttle, steering_angle = action
        throttle = min(max(throttle, 0), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        
        acceleration = self.c_throttle * throttle + friction
        self.joint_speed = self.joint_speed + self.timeSet * acceleration
        if self.joint_speed < 0:
            self.joint_speed = 0

        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client)


    def getHistogram(self,frame,minPer = 0.1,display = False):
        histValue  = np.sum(frame,axis=0)
        maxValue   = np.max(histValue)
        minValue   = minPer * maxValue
        indexArray = np.where(histValue >= minValue)
        basePoint  = int(np.average(indexArray))
        return basePoint

    def cameraFocus(self,position,cameraPos,yaw,distance):
        cameraPos[0] = position[0] + math.cos(yaw) * distance
        cameraPos[1] = position[1] + math.sin(yaw) * distance
        cameraPos[2] = position[2] + 0.2
    
        return cameraPos

    def getFrame(self,cameraPos,orientation):
        rot_mat = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, -2])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            
        view_matrix = p.computeViewMatrix(cameraEyePosition = cameraPos, 
                                          cameraTargetPosition = cameraPos + camera_vec, 
                                          cameraUpVector  = up_vec)

        proj_matrix = p.computeProjectionMatrixFOV(fov=90, 
                                                    aspect=1,
                                                    nearVal=0.001,
                                                    farVal=90)
        frame = p.getCameraImage(32, 32, 
                                view_matrix, 
                                proj_matrix,
                                shadow = False
                                )[2]
            
        frame = np.reshape(frame, (32, 32,4))
        return frame
    
    def get_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.car) 
        orientationv1 = p.getEulerFromQuaternion(orientation)
        cameraPos = [0]*3        
        SetCamera = self.cameraFocus(position,cameraPos,orientationv1[2], distance = 0.5)
        frame = self.getFrame(SetCamera,orientation)
        resized = cv2.resize(frame, (32,32), interpolation = cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hsv_low = np.array([self.H_low, self.S_low, self.V_low], np.uint8)
        hsv_high = np.array([self.H_high, self.S_high, self.V_high], np.uint8)
        mask = cv2.inRange(hsv, hsv_low, hsv_high)/255
        observation = mask
        return observation









