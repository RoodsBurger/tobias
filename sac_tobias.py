from tobias import TobiasController
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import torch
import time
import random
import json
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

        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(8,),  # 4 legs x 2 angles
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20,), dtype=np.float32
        )

        self.steps_in_episode = 0
        self.max_steps_per_episode = 100
        self.initial_pos = None
        self.last_y_position = 0
        self.target_height = 0.1
        self.last_action = None
        self.previous_joint_velocities = None
        
        # Track target positions for each joint
        self.target_positions = {}

    def _setup_robot(self):
        self.plane = p.loadURDF("plane.urdf")
        p.changeDynamics(
            self.plane,
            -1,
            lateralFriction=0.5,
            restitution=0.1,
        )

        robot_start_pos = [0, 0, self.target_height + 0.2]
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

    def _denormalize_action(self, action):
        h_range = self.horizontal_limits[1] - self.horizontal_limits[0]
        v_range = self.vertical_limits[1] - self.vertical_limits[0]

        h_actions = action[::2]
        v_actions = action[1::2]
        h_angles = h_range * (h_actions + 1) / 2 + self.horizontal_limits[0]
        v_angles = v_range * (v_actions + 1) / 2 + self.vertical_limits[0]

        return h_angles, v_angles

    def step(self, action):
        self.steps_in_episode += 1

        action_diff = 0
        if self.last_action is not None:
            action_diff = np.mean(np.abs(action - self.last_action))
        self.last_action = action.copy()

        if self.debug:
            print(f"\nStep {self.steps_in_episode}")
            print(f"Raw action: {action}")

        h_angles, v_angles = self._denormalize_action(action)
        legs = ['fr', 'fl', 'br', 'bl']

        # Set target positions for all joints
        for i, leg in enumerate(legs):
            if self.debug:
                print(f"Moving leg {leg} to horizontal={h_angles[i]:.1f}°, vertical={v_angles[i]:.1f}°")

            # Convert angles to radians for internal tracking
            h_rad = np.radians(h_angles[i])
            v_rad = np.radians(v_angles[i])

            # Store target positions
            self.target_positions[f"{leg}_horizontal"] = h_rad
            self.target_positions[f"{leg}_vertical"] = v_rad

            # Send movement commands
            self.controller.move_leg(leg, h_angles[i], v_angles[i])

        # Wait for movements to complete
        max_wait_steps = 50  # Maximum number of simulation steps to wait
        steps_waited = 0

        while not self._all_movements_complete() and steps_waited < max_wait_steps:
            p.stepSimulation()
            steps_waited += 1

            if self.render_mode == "human":
                self._render_frame()

        observation = self._get_observation()
        reward = self._compute_reward(observation, action_diff)
        terminated = self._check_termination(observation)
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot)
        base_orient = p.getEulerFromQuaternion(base_orient)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot)

        joint_velocities = []
        for leg in ['fr', 'fl', 'br', 'bl']:
            for category in ['horizontal', 'vertical']:
                joint_state = p.getJointState(self.robot,
                                           self.controller.joint_indices[category][leg])
                joint_velocities.append(joint_state[1])

        self.previous_joint_velocities = np.array(joint_velocities)

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

    def _compute_reward(self, observation, action_diff):
        base_pos = observation[0:3]
        base_orient = observation[3:6]
        base_vel = observation[6:9]
        base_angular_vel = observation[9:12]

        forward_velocity = -base_vel[1]
        forward_velocity_reward = np.clip(forward_velocity * 20.0, -50.0, 50.0)

        distance_moved = self.last_y_position - base_pos[1]
        self.last_y_position = base_pos[1]
        progress_reward = distance_moved * 15.0

        height_error = abs(base_pos[2] - self.target_height)
        height_reward = -height_error * 1.0

        orientation_penalty = -(abs(base_orient[0]) + abs(base_orient[1])) * 8.0

        energy_penalty = -0.01 * np.sum(np.square(self.previous_joint_velocities)) if self.previous_joint_velocities is not None else 0

        angular_velocity_penalty = -0.1 * np.sum(np.square(base_angular_vel))
        action_smoothness_reward = -3.0 * action_diff

        stability_reward = 5.0 if (height_error < 0.05 and
                                 abs(base_orient[0]) < 0.2 and
                                 abs(base_orient[1]) < 0.2) else 0.0

        alive_bonus = 1.0 + max(0, forward_velocity) 

        total_reward = (
            forward_velocity_reward * 1.0 +
            progress_reward * 1.0 +
            height_reward * 0.8 +
            orientation_penalty * 0.6 +
            energy_penalty * 0.3 +
            angular_velocity_penalty * 0.4 +
            stability_reward * 1.0 +
            alive_bonus * 0.5 +
            action_smoothness_reward * 0.8
        )

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

        return terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_in_episode = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        self._setup_robot()
        self.controller.default_stance(n_steps=100)

        self.last_action = None
        self.previous_joint_velocities = None
        self.target_positions = {}  # Reset target positions

        observation = self._get_observation()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def _render_frame(self):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    def close(self):
        p.disconnect()

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # Added this import
from torch.distributions import Normal
import os
import random
import numpy as np
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm

os.makedirs("sac_checkpoints", exist_ok=True)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Linear(hidden_dim // 2, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        mean = torch.tanh(mean)

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x = normal.rsample()

        action = torch.tanh(x)

        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def train_quadruped(render: bool = False, debug: bool = False):
    env = QuadrupedEnv(render_mode="human" if render else None, debug=debug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total_timesteps = 1000000
    learning_starts = 10000
    batch_size = 512
    tau = 0.001
    gamma = 0.99
    alpha = 0.2
    actor_lr = 1e-4
    critic_lr = 3e-4
    buffer_size = 1000000
    grad_clip = 1.0

    last_checkpoint = 0
    checkpoint_interval = 10000

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(device)
    critic1 = Critic(state_dim, action_dim).to(device)
    critic2 = Critic(state_dim, action_dim).to(device)
    critic1_target = Critic(state_dim, action_dim).to(device)
    critic2_target = Critic(state_dim, action_dim).to(device)

    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=critic_lr)

    replay_buffer = ReplayBuffer(capacity=buffer_size)

    actor_scheduler = optim.lr_scheduler.StepLR(actor_optimizer, step_size=100000, gamma=0.95)
    critic1_scheduler = optim.lr_scheduler.StepLR(critic1_optimizer, step_size=100000, gamma=0.95)
    critic2_scheduler = optim.lr_scheduler.StepLR(critic2_optimizer, step_size=100000, gamma=0.95)

    logger = configure("./sac_tensorboard/", ["tensorboard"])
    progress_bar = tqdm(total=total_timesteps, desc="Training")

    try:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_num = 0
        total_steps = 0

        exploration_noise = 0.3
        min_exploration_noise = 0.05
        noise_decay = 0.999995

        while total_steps < total_timesteps:
            if total_steps < learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action, _ = actor.sample(state_tensor)
                    action = action.cpu().numpy().flatten()
                    noise = np.random.normal(0, exploration_noise, size=action.shape)
                    action = np.clip(action + noise, -1, 1)
                    exploration_noise = max(min_exploration_noise,
                                         exploration_noise * noise_decay)

            next_state, reward, done, truncated, _ = env.step(action)
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            progress_bar.update(1)

            replay_buffer.push(state, action, reward, next_state, done)

            if total_steps >= learning_starts and len(replay_buffer) > batch_size:
                for _ in range(2):  # Multiple updates per step
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                        replay_buffer.sample(batch_size)

                    state_batch = torch.FloatTensor(state_batch).to(device)
                    action_batch = torch.FloatTensor(action_batch).to(device)
                    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
                    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
                    done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

                    with torch.no_grad():
                        next_actions, next_log_probs = actor.sample(next_state_batch)
                        target_q1 = critic1_target(next_state_batch, next_actions)
                        target_q2 = critic2_target(next_state_batch, next_actions)
                        target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
                        target_q = reward_batch + (1 - done_batch) * gamma * target_q

                    current_q1 = critic1(state_batch, action_batch)
                    critic1_loss = F.mse_loss(current_q1, target_q)
                    critic1_optimizer.zero_grad()
                    critic1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic1.parameters(), grad_clip)
                    critic1_optimizer.step()

                    current_q2 = critic2(state_batch, action_batch)
                    critic2_loss = F.mse_loss(current_q2, target_q)
                    critic2_optimizer.zero_grad()
                    critic2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic2.parameters(), grad_clip)
                    critic2_optimizer.step()

                    if total_steps % 2 == 0:
                        actions, log_probs = actor.sample(state_batch)
                        q1 = critic1(state_batch, actions)
                        q2 = critic2(state_batch, actions)
                        q = torch.min(q1, q2)
                        actor_loss = (alpha * log_probs - q).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                        actor_optimizer.step()

                        logger.record("train/actor_loss", actor_loss.item())
                        logger.record("train/critic1_loss", critic1_loss.item())
                        logger.record("train/critic2_loss", critic2_loss.item())
                        logger.record("train/q_value", q.mean().item())

                    for target_param, param in zip(critic1_target.parameters(), critic1.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for target_param, param in zip(critic2_target.parameters(), critic2.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            state = next_state

            if done or truncated:
                state, _ = env.reset()
                print(f"Episode {episode_num}: {episode_steps} steps, reward = {episode_reward:.2f}")

                logger.record("train/episode_reward", episode_reward)
                logger.record("train/episode_length", episode_steps)
                logger.record("train/exploration_noise", exploration_noise)
                logger.dump(total_steps)

                actor_scheduler.step()
                critic1_scheduler.step()
                critic2_scheduler.step()

                episode_reward = 0
                episode_steps = 0
                episode_num += 1

                if total_steps - last_checkpoint >= checkpoint_interval:
                    checkpoint_step = (total_steps // checkpoint_interval) * checkpoint_interval
                    try:
                        save_path = os.path.join("sac_checkpoints", f"sac_quadruped_{checkpoint_step}_steps.pth")
                        torch.save({
                            'actor_state_dict': actor.state_dict(),
                            'critic1_state_dict': critic1.state_dict(),
                            'critic2_state_dict': critic2.state_dict(),
                            'critic1_target_state_dict': critic1_target.state_dict(),
                            'critic2_target_state_dict': critic2_target.state_dict(),
                        }, save_path)
                        last_checkpoint = checkpoint_step
                    except Exception as e:
                        print(f"Error saving checkpoint: {str(e)}")

        torch.save({
            'actor_state_dict': actor.state_dict(),
            'critic1_state_dict': critic1.state_dict(),
            'critic2_state_dict': critic2.state_dict(),
            'critic1_target_state_dict': critic1_target.state_dict(),
            'critic2_target_state_dict': critic2_target.state_dict(),
        }, "sac_quadruped_final.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        torch.save({
            'actor_state_dict': actor.state_dict(),
            'critic1_state_dict': critic1.state_dict(),
            'critic2_state_dict': critic2.state_dict(),
            'critic1_target_state_dict': critic1_target.state_dict(),
            'critic2_target_state_dict': critic2_target.state_dict(),
        }, "sac_quadruped_interrupted.pth")

    finally:
        env.close()
        progress_bar.close()

def evaluate_model(model_path: str, render: bool = True, debug: bool = False, save_actions: bool = True):
    print(f"\nEvaluating model from {model_path}")
    env = QuadrupedEnv(render_mode="human" if render else None, debug=debug)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(state_dim, action_dim).to(device)

    checkpoint = torch.load(model_path)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    step_time = (5 * (1/240.0))
    recorded_actions = []

    episodes = 5
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            step_start = time.time()

            with torch.no_grad():
                state = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _ = actor.sample(state)
                action = action.cpu().numpy().flatten()

            positions = {}
            leg_names = ['fr', 'fl', 'br', 'bl']

            if env.debug:
                print(f"\nRaw action: {action}")
                print("Motor angles:")
                for i, leg in enumerate(leg_names):
                    h_action = (action[i*2] + 1) / 2
                    v_action = (action[i*2 + 1] + 1) / 2

                    h_range = env.horizontal_limits[1] - env.horizontal_limits[0]
                    v_range = env.vertical_limits[1] - env.vertical_limits[0]

                    h_angle = env.horizontal_limits[0] + h_action * h_range
                    v_angle = env.vertical_limits[0] + v_action * v_range

                    print(f"{leg}: h={h_angle:.1f}°, v={v_angle:.1f}°")

            for i, leg in enumerate(leg_names):
                h_action = (action[i*2] + 1) / 2
                v_action = (action[i*2 + 1] + 1) / 2

                h_range = env.horizontal_limits[1] - env.horizontal_limits[0]
                v_range = env.vertical_limits[1] - env.vertical_limits[0]

                h_angle = env.horizontal_limits[0] + h_action * h_range
                v_angle = env.vertical_limits[0] + v_action * v_range

                positions[f"{leg}_leg"] = {
                    'horizontal': float(h_angle),
                    'vertical': float(v_angle)
                }

            # Save the positions if this is the first episode
            if episode == 0 and save_actions:
                recorded_actions.append(positions)

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            elapsed = time.time() - step_start
            if elapsed < step_time:
                time.sleep(step_time - elapsed)

            if done or truncated:
                break

        print(f"Episode {episode + 1}: {steps} steps, reward = {total_reward:.2f}")

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
    parser.add_argument('--model_path', default='sac_quadruped_final.pth')
    parser.add_argument('--save_actions', action='store_true', help='Save actions during evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        train_quadruped(render=args.render, debug=args.debug)
    else:
        evaluate_model(args.model_path, render=args.render, debug=args.debug, save_actions=args.save_actions)
