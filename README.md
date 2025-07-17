# Random Maze Solver with PPO and DQN using Reinforcement Learning
This project implements a custom Gymnasium environment for a randomly generated 2D maze and trains Reinforcement Learning (RL) agents using PPO (Proximal Policy Optimization) and DQN (Deep Q-Networks) from the stable-baselines3 library. The agents learn to navigate from a start position to a randomly placed goal while avoiding obstacles.

Requirements

Install dependencies via:
```
pip install -r requirements.txt
```

## Train/Evaluate Agents
```
python main.py
```

This will Train both PPO and DQN agents for 20,000 timesteps each and save models under models/, log training metrics under logs/ppo/ and logs/dqn/ and evaluate after training, you will see average rewards and steps printed per episode. Detailed logs are also saved to TensorBoard. Launch TensorBoard with:

```
tensorboard --logdir logs/
```
 
## Environment Description
RandomMazeEnv  
Grid size: default 5x5  

Obstacles placed randomly with probability wall_prob  

Start: [0, 0], Goal: random, reachable position  

Actions: 0=Up, 1=Down, 2=Left, 3=Right  

Rewards  
Condition	Reward  
Reaching goal	+100  
Max steps hit	-10  
Every step	-1 + 0.01 * (grid_size - manhattan_distance_to_goal)  

## Project Structure

├── main.py                  # Entry point    
├── models/                  # Trained models (saved after training)  
├── logs/                    # TensorBoard logs  
│   ├── ppo_random_maze.zip/  
│   ├── dqn_random_maze.zip/  
│   ├── requirements.txt         # Dependencies  
└── README.md                # Project overview  

## Sample Output
Evaluating PPO for 100 episodes.    
✅ Episode   1: Reward = 98.0, Steps = 6  
✅ Episode   2: Reward = -53.0, Steps = 50  
...

📊 PPO - Mean reward: 23.85 ± 37.22  
📏 PPO - Mean steps : 34.17  
