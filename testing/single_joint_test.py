import pybullet as p
import time
import pybullet_data
from math import sin

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

# Load ground plane and robot
planeId = p.loadURDF("plane.urdf")
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("tobias.urdf", robotStartPos, robotStartOrientation,
                    flags = (
                        p.URDF_USE_INERTIA_FROM_FILE |
                        p.URDF_USE_SELF_COLLISION |
                        p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                    ))

# Get and print joint information
num_joints = p.getNumJoints(robotId)
print(f"\nFound {num_joints} joints:")
print("-" * 50)

# Store joint info for monitoring
joint_limits = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    max_force = joint_info[10]
    max_velocity = joint_info[11]
    
    joint_limits[i] = {
        "name": joint_name,
        "lower": lower_limit,
        "upper": upper_limit
    }
    
    print(f"Joint {i}: {joint_name}")
    print(f"Limits: [{lower_limit:.2f}, {upper_limit:.2f}]")
    print(f"Max Force: {max_force:.2f}")
    print(f"Max Velocity: {max_velocity:.2f}")
    print("-" * 50)

# Which joint are you having trouble with?
problem_joint = 0  # Change this to your joint index

# Increased force and velocity limits for testing
force = 1000  # Increase this if joint isn't reaching target
velocity = 100  # Increase this if movement is too slow

print(f"\nTesting joint {problem_joint} with increased force/velocity...")
print("Press Ctrl+C to exit")

i = 0
try:
    while True:
        target_pos = sin(i*0.01)  # Varies between -1 and 1
        
        # Set joint position with higher force/velocity limits
        p.setJointMotorControl2(robotId, 
                              problem_joint, 
                              controlMode=p.POSITION_CONTROL,
                              targetPosition=target_pos,
                              force=force,
                              maxVelocity=velocity)
        
        # Get actual position
        actual_pos = p.getJointState(robotId, problem_joint)[0]
        
        # Print every 100 steps
        if i % 100 == 0:
            print(f"\nTarget: {target_pos:.3f}")
            print(f"Actual: {actual_pos:.3f}")
            print(f"Limits: [{joint_limits[problem_joint]['lower']:.3f}, {joint_limits[problem_joint]['upper']:.3f}]")
        
        p.stepSimulation()
        time.sleep(1./240.)
        i += 1

except KeyboardInterrupt:
    p.disconnect()
