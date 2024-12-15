import pybullet as p
import time
import pybullet_data
import numpy as np

def setup_simulation():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure debug visualizer
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    
    # Load ground plane and robot
    planeId = p.loadURDF("plane.urdf")
    robotStartPos = [0, 0, 0.5]  # Lower starting position for stability
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    robotId = p.loadURDF("./URDF/tobias.urdf", 
                        robotStartPos, 
                        robotStartOrientation,
                        flags=(p.URDF_USE_INERTIA_FROM_FILE |
                              p.URDF_USE_SELF_COLLISION |
                              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT))
    
    return robotId

def get_joint_index_by_name(robot_id, joint_name):
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode('utf-8') == joint_name:
            return i
    return None

def setup_mimic_constraints(robot_id):
    # Define the leg pairs with their mimic relationships
    leg_pairs = [
        {
            "master": "joint_to_fr",
            "mimic": "fr_foot_joint",
            "ratio": -1.0,  # Negative ratio for opposite movement
            "offset": 0.0
        },
        {
            "master": "joint_to_fl",
            "mimic": "fl_foot_joint",
            "ratio": -1.0,
            "offset": 0.0
        },
        {
            "master": "joint_to_br",
            "mimic": "br_foot_joint",
            "ratio": -1.0,
            "offset": 0.0
        },
        {
            "master": "joint_to_bl",
            "mimic": "bl_foot_joint",
            "ratio": -1.0,
            "offset": 0.0
        }
    ]
    
    constraints = []
    for pair in leg_pairs:
        master_idx = get_joint_index_by_name(robot_id, pair["master"])
        mimic_idx = get_joint_index_by_name(robot_id, pair["mimic"])
        
        # Create constraint
        constraint = p.createConstraint(
            robot_id, master_idx,
            robot_id, mimic_idx,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        
        # Set the gear ratio and maximum force
        p.changeConstraint(
            constraint,
            gearRatio=pair["ratio"],
            erp=0.1,  # Error reduction parameter for constraint stability
            maxForce=1000
        )
        
        constraints.append({
            "constraint_id": constraint,
            "master_joint": master_idx,
            "mimic_joint": mimic_idx,
            "ratio": pair["ratio"]
        })
    
    return constraints

def create_control_sliders(robot_id):
    control_joints = [
        "joint_base_to_fr",
        "joint_base_to_fl",
        "joint_base_to_br",
        "joint_base_to_bl",
        "joint_to_fr",
        "joint_to_fl",
        "joint_to_br",
        "joint_to_bl"
    ]
    
    sliders = []
    for joint_name in control_joints:
        joint_idx = get_joint_index_by_name(robot_id, joint_name)
        if joint_idx is not None:
            joint_info = p.getJointInfo(robot_id, joint_idx)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            # Create slider
            slider = p.addUserDebugParameter(
                paramName=joint_name,
                rangeMin=lower_limit,
                rangeMax=upper_limit,
                startValue=0
            )
            
            sliders.append({
                "joint_idx": joint_idx,
                "slider_id": slider,
                "name": joint_name
            })
            
    return sliders

def main():
    # Setup simulation
    robot_id = setup_simulation()
    
    # Setup constraints and controls
    constraints = setup_mimic_constraints(robot_id)
    sliders = create_control_sliders(robot_id)
    
    # Additional parameters for stable control
    kp = 0.5  # Proportional gain
    kd = 0.1  # Derivative gain
    max_force = 100.0  # Maximum motor force
    
    print("\nControl Interface Ready:")
    print("- Use sliders to control joint positions")
    print("- Base joints control the horizontal leg movement")
    print("- Leg joints control the vertical movement with automatic foot adjustment")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Update joint positions based on sliders
            for control in sliders:
                target_pos = p.readUserDebugParameter(control["slider_id"])
                
                # Get current joint state
                joint_state = p.getJointState(robot_id, control["joint_idx"])
                current_pos = joint_state[0]
                current_vel = joint_state[1]
                
                # Calculate position error
                pos_error = target_pos - current_pos
                
                # Apply position control with PD gains
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=control["joint_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    positionGain=kp,
                    velocityGain=kd,
                    force=max_force
                )
            
            # Step simulation with multiple substeps for stability
            for _ in range(5):
                p.stepSimulation()
            
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\nShutting down simulation...")
        p.disconnect()

if __name__ == "__main__":
    main()
