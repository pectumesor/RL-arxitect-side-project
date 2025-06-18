from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from handler import  load_json_by_name, compute_box_and_boundaries
from constants import HOSPITAL_PATH
from environment import MyMultiGoalEnv
from stable_baselines3.common.env_checker import check_env
import torch

# Create environment
graph_json = load_json_by_name("simple-v1",HOSPITAL_PATH)
graph_data, node_positions, box_size, bounding_box = compute_box_and_boundaries(graph_json["wallGraph"])
env = MyMultiGoalEnv(graph_json["poiConfig"],graph_json["wallGraph"],
                     node_positions, box_size, bounding_box)

check_env(env, warn=True)

# Setup training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent and display a progress bar
model.learn(total_timesteps=1_00_000, progress_bar=True)
# Save the agent
model.save("rl_agents/ppo_nurse_agent")
del model  # delete trained model to demonstrate loading

model = PPO.load("rl_agents/ppo_nurse_agent", env=env)

# Enjoy trained agent
obs, _ = env.reset()
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    print(f"Current state is: {info['state']}")
    env.render("human")
    if done or truncated:
        obs, _ = env.reset()
