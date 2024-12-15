from tobias import TobiasController
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import torch
import time
import os
import json
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Dict, List, Tuple, Optional

class QuadrupedEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode: Optional[str] = None, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.render_mode = render_mode
        
        # Add tolerance for considering a movement complete
        self.position_tolerance = 0.01  # radians
        self.velocity_tolerance = 0.01  # radians/second
        
        # Track target positions for each joint
        self.target_positions = {}

        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)

        self.horizontal_limits = (100, 190)
        self.vertical_limits = (135, 210)

        self.gaits = {
            0: {
                'fr': {'horizontal': 170, 'vertical': 195},
                'bl': {'horizontal': 170, 'vertical': 195},
                'fl': {'horizontal': 120, 'vertical': 155},
                'br': {'horizontal': 120, 'vertical': 155}
            },
            1: {
                'fr': {'horizontal': 120, 'vertical': 155},
                'bl': {'horizontal': 120, 'vertical': 155},
                'fl': {'horizontal': 170, 'vertical': 195},
                'br': {'horizontal': 170, 'vertical': 195}
            },
            2: {
                'fr': {'horizontal': 170, 'vertical': 195},
                'fl': {'horizontal': 145, 'vertical': 165},
                'br': {'horizontal': 145, 'vertical': 165},
                'bl': {'horizontal': 120, 'vertical': 165}
            },
            3: {
                'fr': {'horizontal': 145, 'vertical': 165},
                'fl': {'horizontal': 170, 'vertical': 195},
                'br': {'horizontal': 120, 'vertical': 165},
                'bl': {'horizontal': 145, 'vertical': 165}
            },
            4: {
                'fr': {'horizontal': 120, 'vertical': 165},
                'fl': {'horizontal': 145, 'vertical': 165},
                'br': {'horizontal': 170, 'vertical': 195},
                'bl': {'horizontal': 145, 'vertical': 165}
            },
            5: {
                'fr': {'horizontal': 170, 'vertical': 195},
                'fl': {'horizontal': 120, 'vertical': 165},
                'br': {'horizontal': 170, 'vertical': 195},
                'bl': {'horizontal': 120, 'vertical': 165}
            },
            6: {
                'fr': {'horizontal': 120, 'vertical': 165},
                'fl': {'horizontal': 170, 'vertical': 195},
                'br': {'horizontal': 120, 'vertical': 165},
                'bl': {'horizontal': 170, 'vertical': 195}
            },
            7: {
                'fr': {'horizontal': 170, 'vertical': 195},
                'fl': {'horizontal': 170, 'vertical': 195},
                'br': {'horizontal': 120, 'vertical': 165},
                'bl': {'horizontal': 120, 'vertical': 165}
            },
            8: {
                'fr': {'horizontal': 120, 'vertical': 165},
                'fl': {'horizontal': 120, 'vertical': 165},
                'br': {'horizontal': 170, 'vertical': 195},
                'bl': {'horizontal': 170, 'vertical': 195}
            }
        }

        self.action_space = spaces.Discrete(len(self.gaits))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20,), dtype=np.float32
        )

        self.steps_in_episode = 0
        self.max_steps_per_episode = 1000
        self.initial_pos = None
        self.last_y_position = 0

    def _check_movement_complete(self, leg, category):
        """Check if a specific joint movement is complete."""
        joint_idx = self.controller.joint_indices[category][leg]
        current_state = p.getJointState(self.robot, joint_idx)
        target_pos = self.target_positions.get(f"{leg}_{category}")
        
        if target_pos is None:
            return True
            
        position_error = abs(current_state[0] - target_pos)
        velocity = abs(current_state[1])
        
        return (position_error < self.position_tolerance and 
                velocity < self.velocity_tolerance)

    def _all_movements_complete(self):
        """Check if all joint movements are complete."""
        for leg in ['fr', 'fl', 'br', 'bl']:
            for category in ['horizontal', 'vertical']:
                if not self._check_movement_complete(leg, category):
                    return False
        return True

    def _setup_robot(self):
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane,
            -1,
            lateralFriction=0.5,
            restitution=0.1,
        )

        robot_start_pos = [0, 0, 0.3]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])

        self.robot = p.loadURDF(
            "./URDF/tobias.urdf",
            robot_start_pos,
            robot_start_orientation,
            flags=(p.URDF_USE_INERTIA_FROM_FILE |
                  p.URDF_USE_SELF_COLLISION |
                  p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        )

        self.controller = TobiasController(self.robot)
        self.initial_orientation = robot_start_orientation
        self.initial_pos = robot_start_pos
        self.last_y_position = robot_start_pos[1]

        for i in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_name = joint_info[12].decode('UTF-8')
            if 'foot' in joint_name:
                p.changeDynamics(
                    self.robot,
                    i,
                    lateralFriction=0.5,
                    restitution=0.1
                )

        if self.render_mode == "human":
            self._setup_visualization_axes()

    def _setup_visualization_axes(self):
        axis_length = 1.0
        axis_line_width = 3
        origin = [0, 0, 0]

        p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], axis_line_width)
        p.addUserDebugLine(origin, [-axis_length, 0, 0], [0.5, 0, 0], axis_line_width)
        p.addUserDebugText("+X", [axis_length, 0, 0], [1, 0, 0])
        p.addUserDebugText("-X", [-axis_length, 0, 0], [0.5, 0, 0])

        p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], axis_line_width)
        p.addUserDebugLine(origin, [0, -axis_length, 0], [0, 0.5, 0], axis_line_width)
        p.addUserDebugText("+Y", [0, axis_length, 0], [0, 1, 0])
        p.addUserDebugText("-Y", [0, -axis_length, 0], [0, 0.5, 0])

        p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], axis_line_width)
        p.addUserDebugLine(origin, [0, 0, -axis_length], [0, 0, 0.5], axis_line_width)
        p.addUserDebugText("+Z", [0, 0, axis_length], [0, 0, 1])
        p.addUserDebugText("-Z", [0, 0, -axis_length], [0, 0, 0.5])

    def _get_observation(self):
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        base_orient = p.getEulerFromQuaternion(base_orient)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot)

        joint_angles = []
        for category in ['horizontal', 'vertical']:
            for leg in ['fr', 'fl', 'br', 'bl']:
                joint_state = p.getJointState(self.robot,
                                           self.controller.joint_indices[category][leg])
                joint_angles.append(joint_state[0])

        return np.concatenate([
            base_pos,
            base_orient,
            base_vel,
            base_angular_vel,
            joint_angles
        ]).astype(np.float32)

    def step(self, action):
        self.steps_in_episode += 1

        if self.debug:
            print(f"\nStep {self.steps_in_episode}")
            print(f"Applying gait pattern {action}")

        positions = self.gaits[action]

        if self.debug:
            print("\nApplying leg positions:")
            for leg, angles in positions.items():
                print(f"Moving leg {leg} to horizontal={angles['horizontal']}°, vertical={angles['vertical']}°")

        # Set target positions for all joints
        for leg, angles in positions.items():
            # Convert angles to radians for internal tracking
            h_rad = np.radians(angles['horizontal'])
            v_rad = np.radians(angles['vertical'])

            # Store target positions
            self.target_positions[f"{leg}_horizontal"] = h_rad
            self.target_positions[f"{leg}_vertical"] = v_rad

            # Send movement commands
            self.controller.move_leg(leg, angles['horizontal'], angles['vertical'])

        # Wait for movements to complete
        max_wait_steps = 50  # Maximum number of simulation steps to wait
        steps_waited = 0

        while not self._all_movements_complete() and steps_waited < max_wait_steps:
            p.stepSimulation()
            steps_waited += 1

            if self.render_mode == "human":
                self._render_frame()

        observation = self._get_observation()
        reward = self._compute_reward(observation)
        terminated = self._check_termination(observation)
        truncated = False

        if self.debug:
            base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
            base_vel, base_angular_vel = p.getBaseVelocity(self.robot)
            print(f"\nRobot state after movement:")
            print(f"Position: {base_pos}")
            print(f"Orientation: {p.getEulerFromQuaternion(base_orient)}")
            print(f"Velocity: {base_vel}")
            print(f"Reward: {reward}")

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, {}

    def _compute_reward(self, observation):
        base_pos = observation[0:3]
        base_orient = observation[3:6]
        base_vel = observation[6:9]

        forward_velocity_reward = -base_vel[1] * 10.0
        distance_moved = self.last_y_position - base_pos[1]
        self.last_y_position = base_pos[1]
        progress_reward = distance_moved * 30.0
        height_penalty = -10.0 if abs(base_pos[2] - 0.1) > 0.1 else 0.0
        orientation_penalty = -(abs(base_orient[0]) + abs(base_orient[1])) * 5.0
        alive_bonus = 2.0

        total_reward = (forward_velocity_reward +
                       progress_reward +
                       height_penalty +
                       orientation_penalty +
                       alive_bonus)

        if self.debug and self.steps_in_episode % 100 == 0:
            print(f"\nReward components:")
            print(f"Forward velocity: {forward_velocity_reward:.3f}")
            print(f"Progress: {progress_reward:.3f}")
            print(f"Height penalty: {height_penalty:.3f}")
            print(f"Orientation: {orientation_penalty:.3f}")
            print(f"Total: {total_reward:.3f}")

        return total_reward

    def _check_termination(self, observation):
        base_pos = observation[0:3]
        base_orient = observation[3:6]

        terminated = (
            base_pos[2] < 0.06 or
            abs(base_orient[0]) > 0.8 or
            abs(base_orient[1]) > 0.8 or
            abs(base_pos[0]) > 1.0 or
            self.steps_in_episode >= self.max_steps_per_episode
        )

        if terminated and self.debug:
            print("\nTermination condition met:")
            if base_pos[2] < 0.06:
                print(f"Height too low: {base_pos[2]}")
            if abs(base_orient[0]) > 0.8:
                print(f"Roll too high: {base_orient[0]}")
            if abs(base_orient[1]) > 0.8:
                print(f"Pitch too high: {base_orient[1]}")
            if abs(base_pos[0]) > 1.0:
                print(f"Lateral movement too high: {base_pos[0]}")
            if self.steps_in_episode >= self.max_steps_per_episode:
                print("Max steps reached")

        return terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.debug:
            print("\nResetting environment")

        self.steps_in_episode = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        self._setup_robot()
        self.controller.default_stance(n_steps=100)
        self.target_positions = {}  # Reset target positions

        observation = self._get_observation()

        if self.debug:
            print("Reset complete")

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def _render_frame(self):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    def close(self):
        p.disconnect()

def train_quadruped(render: bool = False, debug: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create checkpoints directory
    os.makedirs("dqn_checkpoints", exist_ok=True)

    env = QuadrupedEnv(render_mode="human" if render else None, debug=debug)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        tensorboard_log="./dqn_tensorboard/",
        device=device,
        policy_kwargs={
            "net_arch": [256, 256]
        }
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./dqn_checkpoints/",
        name_prefix="dqn_quadruped",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    try:
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        model.save("dqn_quadruped_final")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save("dqn_quadruped_interrupted")

    finally:
        env.close()

def evaluate_model(model_path: str, render: bool = True, debug: bool = False, save_actions: bool = True):
    print(f"\nEvaluating model from {model_path}")
    env = QuadrupedEnv(render_mode="human" if render else None, debug=debug)
    model = DQN.load(model_path)
    step_time = (5 * (1/240.0))
    recorded_actions = []  # To store the sequence of actions
    episodes = 5
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        if debug:
            print(f"\nStarting episode {episode + 1}")
            
        # Run exactly 100 steps
        for steps in range(100):
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item())
            
            # Record the gait positions if this is the first episode
            if episode == 0 and save_actions:
                positions = {}
                gait = env.gaits[action]
                for leg, angles in gait.items():
                    positions[f"{leg}_leg"] = {
                        'horizontal': float(angles['horizontal']),
                        'vertical': float(angles['vertical'])
                    }
                recorded_actions.append(positions)
                
            if debug:
                print(f"\nStep {steps + 1}")
                print(f"Selected gait pattern: {action}")
                print("Motor angles:")
                for leg, angles in env.gaits[action].items():
                    print(f"{leg}: h={angles['horizontal']}°, v={angles['vertical']}°")
                    
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Maintain consistent step timing
            elapsed = time.time() - step_start
            if elapsed < step_time:
                time.sleep(step_time - elapsed)
                
        print(f"Episode {episode + 1}: 100 steps, reward = {total_reward:.2f}")
        
        # Save actions after first episode
        if episode == 0 and save_actions:
            print(f"Saving {len(recorded_actions)} actions...")
            with open('recorded_actions.json', 'w') as f:
                json.dump(recorded_actions, f)
            print("Actions saved to recorded_actions.json")
            
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model_path', default='dqn_quadruped_final')
    parser.add_argument('--save_actions', action='store_true', help='Save actions during evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        train_quadruped(render=args.render, debug=args.debug)
    else:
        evaluate_model(args.model_path, render=args.render, debug=args.debug, save_actions=args.save_actions)
