from maze.environment import Maze
from agents.random_agent import RandomAgent

env = Maze(size=5)
agent = RandomAgent()

state = env.reset()
total_reward = 0

for step in range(50):
    action = agent.act()
    state, reward, done = env.step(action)
    total_reward += reward

    print(f"Step {step}: Pos={state}, Reward={reward}")

    if done:
        print("🎉 Goal reached!")
        break

print("Total reward:", total_reward)
