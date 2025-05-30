from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from handler import  load_json_by_name, compute_box_and_boundaries
from constants import HOSPITAL_PATH
from environment import MyMultiGoalEnv
from stable_baselines3.common.env_checker import check_env



# Create environment
graph_json = load_json_by_name("buildingConfig",HOSPITAL_PATH)
graph_data, node_positions, box_size, bounding_box = compute_box_and_boundaries(graph_json["wallGraph"])
env = MyMultiGoalEnv(graph_json["poiConfig"],graph_json["wallGraph"],
                     node_positions, box_size, bounding_box)

check_env(env, warn=True)

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=1000000, progress_bar=True)
# Save the agent
model.save("rl_agents/ppo_nurse_agent")
del model  # delete trained model to demonstrate loading

model = PPO.load("rl_agents/ppo_    nurse_agent", env=env)

# Enjoy trained agent
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render("human")
