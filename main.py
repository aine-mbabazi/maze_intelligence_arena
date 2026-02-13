from maze.environment import Maze
from maze.renderer import render
from agents.q_learning_agent import QLearningAgent
import numpy as np

env = Maze(size=5)
agent = QLearningAgent(size=env.size)

EPISODES = 500

#TRAINING LOOP
for episode in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

    # Optional: decay exploration
    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    if episode % 100 == 0:
        print(f"Episode {episode} complete")

print("Training finished.")

state = env.reset()
total_reward = 0

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
