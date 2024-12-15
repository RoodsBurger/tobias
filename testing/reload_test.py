import pybullet as p
import time
import pybullet_data

def reload_robot():
    robotStartPos = [0,0,1]
    robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
    
    # Remove old robot and sliders but keep the environment
    p.removeAllUserDebugItems()
    robotIds = []
    for i in range(p.getNumBodies()):
        bodyId = i  # In PyBullet, body index is same as body unique id
        bodyInfo = p.getBodyInfo(bodyId)
        if bodyInfo[1].decode('utf-8') != "plane":  # Don't remove the plane
            p.removeBody(bodyId)
            
    # Load new robot
    robotId = p.loadURDF("tobias.urdf", robotStartPos, robotStartOrientation,
                        flags = (
                            p.URDF_USE_INERTIA_FROM_FILE |
                            p.URDF_USE_SELF_COLLISION |
                            p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                        ))

    # Setup sliders and print info
    sliders = []
    num_joints = p.getNumJoints(robotId)
    print(f"\nFound {num_joints} joints:")
    print("-" * 50)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"][joint_info[2]]
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        max_force = joint_info[10]
        max_velocity = joint_info[11]
        
        print(f"Joint {i}: {joint_name}")
        print(f"Type: {joint_type}")
        print(f"Limits: [{lower_limit:.2f}, {upper_limit:.2f}]")
        print(f"Max Force: {max_force:.2f}")
        print(f"Max Velocity: {max_velocity:.2f}")
        print("-" * 50)
        
        # Create slider only for non-fixed joints
        if joint_type != "FIXED":
            slider = p.addUserDebugParameter(
                paramName=f"Joint {i}: {joint_name}",
                rangeMin=lower_limit,
                rangeMax=upper_limit,
                startValue=0
            )
            sliders.append((i, slider))
            
    return robotId, sliders

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

# Load ground plane and robot
planeId = p.loadURDF("plane.urdf")
robotId, sliders = reload_robot()

print("\nUse the sliders in the GUI to control the joints.")
print("Press 'r' to reload the URDF")
print("Press Ctrl+C to exit")

try:
    while True:
        # Check for 'r' key press
        keys = p.getKeyboardEvents()
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            print("\nReloading URDF...")
            robotId, sliders = reload_robot()
        
        # Update joint positions based on sliders
        for joint_idx, slider_id in sliders:
            target_pos = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(robotId,
                                  joint_idx,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=target_pos)
        
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()
