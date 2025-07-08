from stable_baselines3 import PPO
from handler import  load_json_by_name, compute_box_and_boundaries
from constants import HOSPITAL_PATH, TRAINING_STEPS
from environment import MyMultiGoalEnv

# Create environment
graph_json = load_json_by_name("oneRoom",HOSPITAL_PATH)
graph_data, node_positions, box_size, bounding_box = compute_box_and_boundaries(graph_json["wallGraph"])

env = MyMultiGoalEnv(graph_json["poiConfig"],graph_json["wallGraph"],
                    node_positions, box_size, bounding_box, render_mode="human")

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
   verbose=1,
)
# Train the agent and display a progress bar
model.learn(total_timesteps=TRAINING_STEPS, progress_bar=True)
# Save the agent
model.save(f"rl_agents/ppo_nurse_agent_{1}")
del model  # delete trained model to demonstrate loading

model = PPO.load(f"rl_agents/ppo_nurse_agent_{1}", env=env)

obs, info = env.reset()

for _ in range(10000):
    action,_ = model.predict(obs)
    obs, reward, done,_, info = env.step(action)

    print(f"Observation: {obs} and Action: {action} and info: {info}")

    if done:
        obs = env.reset()
    env.render()
