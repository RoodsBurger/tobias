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

# Store joint limits
joint_limits = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"][joint_info[2]]
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    max_force = joint_info[10]
    max_velocity = joint_info[11]
    
    joint_limits.append({
        'lower': lower_limit,
        'upper': upper_limit,
        'name': joint_name
    })
    
    print(f"Joint {i}: {joint_name}")
    print(f"Type: {joint_type}")
    print(f"Limits: [{lower_limit:.2f}, {upper_limit:.2f}]")
    print(f"Max Force: {max_force:.2f}")
    print(f"Max Velocity: {max_velocity:.2f}")
    print("-" * 50)

# Control mode
mode = p.POSITION_CONTROL

# Main simulation loop
i = 0
try:
    while True:
        # Move each joint through its full range using sine wave scaled to joint limits
        for joint in range(num_joints):
            limits = joint_limits[joint]
            
            # Scale sine wave to joint limits
            limit_range = limits['upper'] - limits['lower']
            mid_point = (limits['upper'] + limits['lower']) / 2
            # Sine wave oscillates between -1 and 1, so multiply by half the range
            # and add to mid point to oscillate between limits
            target = mid_point + sin(i*0.01) * (limit_range/2)
            
            # Print every 100 steps
            if i % 100 == 0:
                current_pos = p.getJointState(robotId, joint)[0]
                print(f"Joint {joint} ({limits['name']})")
                print(f"Target: {target:.2f}, Current: {current_pos:.2f}")
                print(f"Limits: [{limits['lower']:.2f}, {limits['upper']:.2f}]")
                print("-" * 30)
            
            p.setJointMotorControl2(robotId, joint,
                                  controlMode=mode,
                                  targetPosition=target,
                                  maxVelocity=5.0)  # Adjusted for smoother motion
        
        p.stepSimulation()
        time.sleep(1./240.)
        i += 1

except KeyboardInterrupt:
    p.disconnect()
