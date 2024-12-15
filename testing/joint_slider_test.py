import pybullet as p
import time
import pybullet_data
import numpy as np

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

# Load ground plane and robot
planeId = p.loadURDF("plane.urdf")
robotStartPos = [0,0,1]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("./URDF/tobias.urdf", robotStartPos, robotStartOrientation,
                   flags = (
                       p.URDF_USE_INERTIA_FROM_FILE |
                       p.URDF_USE_SELF_COLLISION |
                       p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                   ))

def get_link_index_by_name(robotId, link_name):
   num_joints = p.getNumJoints(robotId)
   for i in range(num_joints):
       joint_info = p.getJointInfo(robotId, i)
       child_link_name = joint_info[12].decode('UTF-8')
       if child_link_name == link_name:
           return joint_info[0]
   return None

def get_joint_index_by_name(robot_id, joint_name):
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode('utf-8') == joint_name:  # Joint name is at index 1
            return i
    return None

# Define all master-mimic joint pairs
leg_pairs = [
    ("joint_to_fr", "fr_foot_joint"),
    ("joint_to_fl", "fl_foot_joint"),
    ("joint_to_br", "br_foot_joint"),
    ("joint_to_bl", "bl_foot_joint")
]

# Get all joint indices
joint_pairs = []
for master_name, mimic_name in leg_pairs:
    master_joint = get_joint_index_by_name(robotId, master_name)
    mimic_joint = get_joint_index_by_name(robotId, mimic_name)
    joint_pairs.append((master_joint, mimic_joint))

# Get the link indices
fr_connector_index = get_link_index_by_name(robotId, 'fr_connector')
fr_foot_index = get_link_index_by_name(robotId, 'fr_foot')

# Find the connection points and create sliders
num_joints = p.getNumJoints(robotId)
sliders = []
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
   
   if joint_type != "FIXED" and not joint_name.endswith('foot_joint'): 
       slider = p.addUserDebugParameter(
           paramName=f"Joint {i}: {joint_name}",
           rangeMin=lower_limit,
           rangeMax=upper_limit,
           startValue=0
       )
       sliders.append((i, slider))

print("\nUse the sliders in the GUI to control the joints.")
print("Press Ctrl+C to exit")

try:
    while True:
        # Update joint positions based on sliders
        for joint_idx, slider_id in sliders:
            target_pos = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(robotId,
                                  joint_idx,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=target_pos)
            
            # Control all mimic joints when their master moves
            for master, mimic in joint_pairs:
                if joint_idx == master:
                    p.setJointMotorControl2(robotId,
                                          mimic,
                                          controlMode=p.POSITION_CONTROL,
                                          targetPosition=target_pos)
        
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
   p.disconnect()
