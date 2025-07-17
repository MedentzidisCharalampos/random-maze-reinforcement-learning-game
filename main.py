import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import pygame
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
import os
from torch.utils.tensorboard import SummaryWriter
class RandomMazeEnv(gym.Env):
    def __init__(self, grid_size=5, wall_prob=0.3):
        super().__init__()
        self.grid_size = grid_size
        self.wall_prob = wall_prob
        self.max_steps = 50
        self.start_pos = [0, 0]
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.float32
        )

        self.maze = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos.copy()
        self.steps = 0

        while True:
            self._generate_maze()
            if self._is_solvable():
                break

        return np.array(self.agent_pos, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x, new_y = x + dx, y + dy

        if self._valid(new_x, new_y):
            self.agent_pos = [new_x, new_y]

        # Check termination and truncation conditions
        terminated = self.agent_pos == self.goal_pos
        truncated = self.steps >= self.max_steps

        # Reward logic
        if terminated:
            reward = 100
        elif truncated:
            reward = -10
        else:
            reward = -1 + 0.01 * (self.grid_size - self._manhattan_to_goal())

        obs = np.array(self.agent_pos, dtype=np.float32)
        info = {}

        return obs, reward, terminated, truncated, info

    def _manhattan_to_goal(self):
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

    def _valid(self, x, y):
        return (
            0 <= x < self.grid_size
            and 0 <= y < self.grid_size
            and self.maze[x, y] != 1
        )

    def _generate_maze(self):
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=int)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if [x, y] != self.start_pos and random.random() < self.wall_prob:
                    self.maze[x, y] = 1

        free_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if self.maze[x, y] == 0 and [x, y] != self.start_pos
        ]
        self.goal_pos = list(random.choice(free_cells))
        self.maze[self.goal_pos[0], self.goal_pos[1]] = 2

    def _is_solvable(self):
        queue = deque()
        visited = set()
        queue.append(tuple(self.start_pos))
        visited.add(tuple(self.start_pos))

        while queue:
            cx, cy = queue.popleft()
            if [cx, cy] == self.goal_pos:
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < self.grid_size
                    and 0 <= ny < self.grid_size
                    and (nx, ny) not in visited
                    and self.maze[nx, ny] != 1
                ):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        return False

    def close(self):
        pass

            
def train_model(algo: str, total_timesteps=20000):
    env = Monitor(RandomMazeEnv())
    log_dir = f"logs/{algo}"
    os.makedirs(log_dir, exist_ok=True)

    if algo == "ppo":
       model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,          # Faster adaptation
        n_steps=256,                 # More unrolls per update
        batch_size=64,
        n_epochs=10,
        gamma=0.995,                 # Slightly more future-oriented
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,               # More exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
    )
    elif algo == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=64,
            gamma=0.995,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
            max_grad_norm=10,
            verbose=1,
            tensorboard_log=log_dir,
        )


    else:
        raise ValueError("Unsupported algorithm")

    print(f"\n🧠 Training {algo.upper()}...")
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{algo}_random_maze")
    env.close()


def evaluate_model(algo: str, n_episodes=100):
    model_path = f"models/{algo}_random_maze"
    log_dir = f"logs/{algo}_eval"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "dqn":
        model = DQN.load(model_path)
    else:
        raise ValueError("Unsupported algorithm")

    rewards = []
    steps_per_episode = []

    print(f"\n🎯 Evaluating {algo.upper()} for {n_episodes} episodes. \n")

    for ep in range(n_episodes):
        env = RandomMazeEnv()
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        writer.add_scalar("Reward", total_reward, ep)
        writer.add_scalar("Steps", step_count, ep)

        print(f"  ✅ Episode {ep+1:>3}: Reward = {total_reward}, Steps = {step_count}")
        rewards.append(total_reward)
        steps_per_episode.append(step_count)
        env.close()

    writer.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps_per_episode)

    print(f"\n📊 {algo.upper()} - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"📏 {algo.upper()} - Mean steps : {mean_steps:.2f}")

if __name__ == "__main__":
    for algo in ["ppo", "dqn"]:
        train_model(algo, total_timesteps=20000)
        evaluate_model(algo, n_episodes=100)

        

