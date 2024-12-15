import pybullet as p
import time
import pybullet_data
from math import sin

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("myrobot.urdf", robotStartPos, robotStartOrientation,
                    flags = (
                        p.URDF_USE_INERTIA_FROM_FILE |
                        p.URDF_USE_SELF_COLLISION |
                        p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                    ))
mode = p.POSITION_CONTROL
jointIndex = 0 # first joint is number 0
for i in range (10000):
    p.setJointMotorControl2(robotId, 0, controlMode=mode, targetPosition=sin(i*0.01))
    p.setJointMotorControl2(robotId, 1, controlMode=mode, targetPosition=sin(i*0.01))
    p.setJointMotorControl2(robotId, 2, controlMode=mode, targetPosition=sin(i*0.01))
    p.setJointMotorControl2(robotId, 3, controlMode=mode, targetPosition=sin(i*0.01))
    p.setJointMotorControl2(robotId, 4, controlMode=mode, targetPosition=sin(i*0.01))
    p.stepSimulation()
    time.sleep(1./240.)
robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
print(robotPos, robotOrn)
p.disconnect()
