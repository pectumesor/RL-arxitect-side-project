from handler import  load_json_by_name, compute_box_and_boundaries
from constants import HOSPITAL_PATH
from environment import MyMultiGoalEnv
import numpy as np
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create environment
graph_json = load_json_by_name("simple-v1",HOSPITAL_PATH)
graph_data, node_positions, box_size, bounding_box = compute_box_and_boundaries(graph_json["wallGraph"])
env = DummyVecEnv([lambda: MyMultiGoalEnv(graph_json["poiConfig"],graph_json["wallGraph"],
                     node_positions, box_size, bounding_box, render_mode="human")])

# Normalize input features and rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Instantiate the agent
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1,
                     learning_rate=0.0001, ent_coef=0.05)
# Train the agent and display a progress bar
#model.learn(total_timesteps=1_000_000, progress_bar=True)
# Save the agent
#model.save(f"rl_agents/ppo_lstm_nurse_agent_{0}")
del model  # delete trained model to demonstrate loading

model = RecurrentPPO.load(f"rl_agents/ppo_lstm_nurse_agent_{0}", env=env)

obs = env.reset()
lstm_states = None
episode_starts = np.ones((env.num_envs,), dtype=bool)

for _ in range(10000):
    action, lstm_states = model.predict(obs,state=lstm_states,episode_start=episode_starts,deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_starts = done  # Important: update episode starts based on termination

    print(f"Observation: {obs} and Action: {action} and info: {info}")
    if done:
        obs = env.reset()
    env.render()

