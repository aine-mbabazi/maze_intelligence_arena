

from maze.environment import Maze
from maze.renderer import render
from agents.random_agent import RandomAgent
from agents.bfs_agent import BFSAgent
import time

env = Maze(size=5)

# Toggle agent here
# agent = RandomAgent()
agent = BFSAgent(env)

env.reset()
total_reward = 0

for step in range(50):
    render(env)

    action = agent.act()
    state, reward, done = env.step(action)
    total_reward += reward

    time.sleep(0.3)

    if done:
        render(env)
        print("🎉 Goal reached!")
        break

print("Total reward:", total_reward)

