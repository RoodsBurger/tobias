import pybullet as p
import time
import pybullet_data
import numpy as np

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(1)

# Configure debug visualizer
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# Load ground plane with improved physics
planeId = p.loadURDF("plane.urdf")
p.changeDynamics(
    planeId, 
    -1,
    lateralFriction=0.5,
    restitution=0.1
)

# Load robot with lower starting position
robotStartPos = [0, 0, 0.3]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("./URDF/tobias.urdf", 
                    robotStartPos, 
                    robotStartOrientation,
                    flags=(p.URDF_USE_INERTIA_FROM_FILE |
                          p.URDF_USE_SELF_COLLISION |
                          p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT))

# Add coordinate axes
def add_coordinate_axes():
    length = 0.5
    p.addUserDebugLine([0, 0, 0], [length, 0, 0], [1, 0, 0], 2)  # X axis - red
    p.addUserDebugText("X", [length, 0, 0], [1, 0, 0])
    p.addUserDebugLine([0, 0, 0], [0, length, 0], [0, 1, 0], 2)  # Y axis - green
    p.addUserDebugText("Y", [0, length, 0], [0, 1, 0])
    p.addUserDebugLine([0, 0, 0], [0, 0, length], [0, 0, 1], 2)  # Z axis - blue
    p.addUserDebugText("Z", [0, 0, length], [0, 0, 1])

add_coordinate_axes()

def get_joint_index_by_name(robot_id, joint_name):
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode('utf-8') == joint_name:
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

# Create sliders
sliders = []
for i in range(p.getNumJoints(robotId)):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    if not joint_name.endswith('foot_joint'):
        slider = p.addUserDebugParameter(
            paramName=f"{joint_name}",
            rangeMin=joint_info[8],
            rangeMax=joint_info[9],
            startValue=0
        )
        sliders.append((i, slider))

# Create text items for state monitoring
text_start_x = 1.5
debug_texts = []

def create_debug_texts():
    global debug_texts
    y_offset = 0
    text_spacing = 0.05
    
    texts = [
        "Position (XYZ):", "Orientation (RPY):", "Velocity (XYZ):", 
        "Angular Velocity:", "Forward Velocity Reward:", "Height Reward:",
        "Height Diff:", "Orientation Penalty:", "Angular Velocity Penalty:",
        "Lateral Movement Penalty:", "Energy Penalty:", "Alive Bonus:",
        "Progress:", "Total Reward:"
    ]
    
    for i, text in enumerate(texts):
        item_id = p.addUserDebugText(
            text,
            [text_start_x, 0, 1.0 - (i * text_spacing)],
            [1, 1, 1],
        )
        debug_texts.append(item_id)

create_debug_texts()

def get_joint_angles():
    angles = []
    for joint_idx, _ in sliders:
        angles.append(p.getJointState(robotId, joint_idx)[0])
    return angles

def update_state_info():
    global debug_texts
    base_pos, base_orient = p.getBasePositionAndOrientation(robotId)
    base_orient = p.getEulerFromQuaternion(base_orient)
    base_vel, base_angular_vel = p.getBaseVelocity(robotId)
    
    # Calculate all rewards and metrics
    forward_velocity_reward = -base_vel[1] * 2.0
    target_height = 0.05
    height_reward = base_pos[2]
    orientation_penalty = -5.0 * (base_orient[0]**2 + base_orient[1]**2)
    angular_velocity_penalty = -0.5 * np.sum(np.square(base_angular_vel))
    lateral_movement_penalty = -0.5 * base_pos[0]**2
    
    joint_angles = get_joint_angles()
    energy_penalty = -0.001 * np.sum(np.square(joint_angles))
    
    alive_bonus = 0.5
    distance_moved = -(base_pos[1] - robotStartPos[1])
    progress_bonus = 2.0 * distance_moved if distance_moved > 0 else 0.0
    
    total_reward = (forward_velocity_reward + height_reward + 
                   orientation_penalty + angular_velocity_penalty +
                   lateral_movement_penalty + energy_penalty +
                   alive_bonus + progress_bonus)

    # Update all debug texts
    texts = [
        f"Position: {base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}",
        f"Orientation: {base_orient[0]:.3f}, {base_orient[1]:.3f}, {base_orient[2]:.3f}",
        f"Velocity: {base_vel[0]:.3f}, {base_vel[1]:.3f}, {base_vel[2]:.3f}",
        f"Angular Velocity: {np.linalg.norm(base_angular_vel):.3f}",
        f"Forward Velocity Reward: {forward_velocity_reward:.3f}",
        f"Height: {height_reward:.3f}",
        f"Orientation Penalty: {orientation_penalty:.3f}",
        f"Angular Velocity Penalty: {angular_velocity_penalty:.3f}",
        f"Lateral Movement Penalty: {lateral_movement_penalty:.3f}",
        f"Energy Penalty: {energy_penalty:.3f}",
        f"Alive Bonus: {alive_bonus:.3f}",
        f"Progress Bonus: {progress_bonus:.3f}",
        f"Total Reward: {total_reward:.3f}"
    ]
    
    for i, (text_id, new_text) in enumerate(zip(debug_texts, texts)):
        p.removeUserDebugItem(text_id)
        debug_texts[i] = p.addUserDebugText(
            new_text,
            [text_start_x, 0, 1.0 - (i * 0.05)],
            [1, 1, 1]
        )

print("\nUse the sliders to control the joints.")
print("Press Ctrl+C to exit")

try:
    while True:
        # Update joints from sliders
        for joint_idx, slider_id in sliders:
            target_pos = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(
                robotId,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=1.0,
                maxVelocity=2.8
            )
            
            # Update mimic joints
            for master, mimic in joint_pairs:
                if joint_idx == master:
                    p.setJointMotorControl2(
                        robotId,
                        mimic,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=1.0,
                        maxVelocity=2.8
                    )
        
        update_state_info()
        time.sleep(1./240.)

except KeyboardInterrupt:
    p.disconnect()
