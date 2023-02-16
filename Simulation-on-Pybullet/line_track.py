import pybullet as p
import os

class track:
    def __init__(self, client,option = 0):
        if option == 0 :
            f_name = os.path.join(os.path.dirname(__file__), 'Fix_asemblyv2/urdf/Fix_asemblyv2.urdf')
            p.loadURDF(fileName=f_name,
                    basePosition=[0, 0, 0.01],
                    physicsClientId=client,
                    useFixedBase = True)
            
        if option == 1 :
            f_name = os.path.join(os.path.dirname(__file__), "Fix_asemblyv3/urdf/Fix_asemblyv3.urdf")
            p.loadURDF(fileName=f_name,
                    basePosition=[0, 0, 0.01],
                    physicsClientId=client,
                    useFixedBase = True)


