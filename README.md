### Reinforcement Learning Side Project on Arxitect agents

I am working, with support of a PHD Student from IVIA Lab, as a side project on managing to train the arxitect agents with RL,
such that their movements aren't random.

### Tech stack:
 
- Gymnasium Library
- Stable Baselines for PPO/DQN Algorithms
- to be continued...

### Disclaimer: This is an ongoing project

Progress thus far:

- Decided to a single agent environment, training only a single nurse
- Decided to keep a dedicated class for the Agent, due to the Finite State Machine
- Managed to train the agent using Stable Baselines PPO Algorithm
- Due to the complicated agent step function, often PPO finds the best reward by just not moving at all.

Next steps:

- Adjust step function to be much simpler (No collision checks)
- Add huge penalty to the reward function on collison (Let PPO handle this)
- Adjust Environment observation space to be the Lidar readings
- Create other Environments for the other Agents
- Procceed on a MultiAgent approach
