from stable_baselines3 import PPO, A2C
from handler import  load_json_by_name, compute_box_and_boundaries
from constants import HOSPITAL_PATH
from environment import MyMultiGoalEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create environment
graph_json = load_json_by_name("simple-v1",HOSPITAL_PATH)
graph_data, node_positions, box_size, bounding_box = compute_box_and_boundaries(graph_json["wallGraph"])

env = MyMultiGoalEnv(graph_json["poiConfig"],graph_json["wallGraph"],
                    node_positions, box_size, bounding_box, render_mode="human")

venv = DummyVecEnv([lambda: env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, training=True)

# Instantiate the agent
model = A2C(
    "MlpPolicy", venv,
    verbose=1,
    ent_coef=0.05,
     )

# Train the agent and display a progress bar
model.learn(total_timesteps=1_000_000, progress_bar=True)
# Save the agent
model.save(f"rl_agents/a2c_nurse_agent_{0}")
del model  # delete trained model to demonstrate loading

model = PPO.load(f"rl_agents/a2c_nurse_agent_{0}", env=venv)

obs = venv.reset()

for _ in range(10000):
    action,_ = model.predict(obs)
    obs, reward, done, info = venv.step(action)

   # print(f"Observation: {obs} and Action: {action} and info: {info}")

    if done:
        obs = venv.reset()
    venv.render()
