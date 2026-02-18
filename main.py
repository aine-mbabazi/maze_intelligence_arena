from maze.environment import Maze
from maze.renderer import render
from agents.q_learning_agent import QLearningAgent
import numpy as np
from analysis.metrics import save_rewards

env = Maze(size=5)
agent = QLearningAgent(size=env.size)

EPISODES = 500

#TRAINING LOOP
EPISODES = 1000
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

save_rewards(episode_rewards)
print("Training finished and rewards saved.")


for step in range(50):
    render(env)
    action = agent.act(state)
    state, reward, done = env.step(action)
    total_reward += reward

    if done:
        render(env)
        print("🎉 Goal reached!")
        break

print("Total reward:", total_reward)

agent.save("models/q_table.npy")

