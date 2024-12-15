import pybullet as p
import numpy as np
import time
import pybullet_data
from typing import Dict, List, Tuple

class TobiasController:
    def __init__(self, robot_id: int):
        self.robot_id = robot_id

        self.horizontal_zero = 100
        self.vertical_zero = 150

        self.joint_damping = 0.1  # Reduced damping

        self.joint_states = {
            'horizontal': {
                'fr': 'joint_base_to_fr',
                'fl': 'joint_base_to_fl',
                'br': 'joint_base_to_br',
                'bl': 'joint_base_to_bl'
            },
            'vertical': {
                'fr': 'joint_to_fr',
                'fl': 'joint_to_fl',
                'br': 'joint_to_br',
                'bl': 'joint_to_bl'
            },
            'foot': {
                'fr': 'fr_foot_joint',
                'fl': 'fl_foot_joint',
                'br': 'br_foot_joint',
                'bl': 'bl_foot_joint'
            }
        }

        self.joint_indices = self._get_joint_indices()
        self._configure_joints()

    def _configure_joints(self):
        for category in self.joint_indices.values():
            for joint_idx in category.values():
                p.changeDynamics(
                    self.robot_id,
                    joint_idx,
                    jointDamping=self.joint_damping
                )

    def _get_joint_index(self, joint_name: str) -> int:
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[1].decode('utf-8') == joint_name:
                return i
        return None

    def _get_joint_indices(self) -> Dict:
        indices = {}
        for category, joints in self.joint_states.items():
            indices[category] = {}
            for leg, joint_name in joints.items():
                indices[category][leg] = self._get_joint_index(joint_name)
        return indices

    def move_leg(self, leg: str, horizontal_angle: float, vertical_angle: float, force: float = 1.667, motor_speed: float = 2.8):
        horizontal_rad = np.radians((horizontal_angle - self.horizontal_zero))
        vertical_rad = np.radians((vertical_angle - self.vertical_zero))
        
        p.setJointMotorControl2(
            self.robot_id,
            self.joint_indices['horizontal'][leg],
            p.POSITION_CONTROL,
            targetPosition=horizontal_rad,
            force=force,
            maxVelocity=motor_speed
        )
        
        for joint_type in ['vertical', 'foot']:
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[joint_type][leg],
                p.POSITION_CONTROL,
                targetPosition=vertical_rad,
                force=force,
                maxVelocity=motor_speed
            )

    def default_stance(self, n_steps: int = 100):
        positions = {
            'fr': {'horizontal': 150, 'vertical': 170},
            'fl': {'horizontal': 150, 'vertical': 170},
            'br': {'horizontal': 150, 'vertical': 170},
            'bl': {'horizontal': 150, 'vertical': 170}
        }

        for leg, angles in positions.items():
            self.move_leg(leg, angles['horizontal'], angles['vertical'], force=1.0)
        
        for _ in range(n_steps):
            p.stepSimulation()
            time.sleep(1./240.)

    def walk(self, steps: int = 5, step_size: float = 20, step_height: float = 10):
        base_h = 150  # Base horizontal angle
        base_v = 170  # Base vertical angle

        for i in range(steps):
            print(f"\nTaking step {i}...")

            positions = {
                'fl': {'horizontal': base_h + step_size, 'vertical': base_v + step_height},
                'br': {'horizontal': base_h - step_size, 'vertical': base_v + step_height}
            }

            for leg, angles in positions.items():
                self.move_leg(leg, angles['horizontal'], angles['vertical'])

            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./240.)

            positions = {
                'fl': {'horizontal': base_h, 'vertical': base_v - step_height},
                'br': {'horizontal': base_h, 'vertical': base_v - step_height}
            }

            for leg, angles in positions.items():
                self.move_leg(leg, angles['horizontal'], angles['vertical'])

            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./240.)

            self.default_stance()

            positions = {
                'fr': {'horizontal': base_h + step_size, 'vertical': base_v + step_height},
                'bl': {'horizontal': base_h - step_size, 'vertical': base_v + step_height}
            }

            for leg, angles in positions.items():
                self.move_leg(leg, angles['horizontal'], angles['vertical'])

            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./240.)

            # Lower legs
            positions = {
                'fr': {'horizontal': base_h, 'vertical': base_v - step_height},
                'bl': {'horizontal': base_h, 'vertical': base_v - step_height}
            }

            for leg, angles in positions.items():
                self.move_leg(leg, angles['horizontal'], angles['vertical'])

            for _ in range(50):
                p.stepSimulation()
                time.sleep(1./240.)

            self.default_stance()

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # Load ground plane with reduced friction
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(
        planeId, 
        -1,
        lateralFriction=0.5,
        restitution=0.1,      
    )
    
    robotStartPos = [0, 0, 0.3]  
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, np.pi])
    
    robot = p.loadURDF(
        "./URDF/tobias.urdf",
        robotStartPos,
        robotStartOrientation,
        flags=(p.URDF_USE_INERTIA_FROM_FILE |
               p.URDF_USE_SELF_COLLISION |
               p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    )
    
    controller = TobiasController(robot)
    
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        joint_name = joint_info[12].decode('UTF-8')
        if 'foot' in joint_name:
            p.changeDynamics(
                robot,
                i,
                lateralFriction=0.5,
                restitution=0.1
            )
    
    try:
        print("Settling robot...")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        
        print("Moving to default stance...")
        controller.default_stance(100)
        
        print("Starting to walk...")
        controller.walk(steps=5)
        
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
        
    except KeyboardInterrupt:
        p.disconnect()
