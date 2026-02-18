import numpy as np
from environment import MazeEnv
from agents.random_agent import RandomAgent
from agents.bfs_agent import BFSAgent
from agents.q_learning_agent import QLearningAgent


def evaluate(agent, episodes=50):
    env = MazeEnv()
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


if __name__ == "__main__":
    env = MazeEnv()

    random_agent = RandomAgent(env)
    bfs_agent = BFSAgent(env)

    q_agent = QLearningAgent(
        state_size=env.state_size,
        action_size=4
    )
    q_agent.load("models/q_table.npy")
    q_agent.epsilon = 0.0  

    agents = {
        "Random": random_agent,
        "BFS": bfs_agent,
        "Q-Learning": q_agent
    }

    for name, agent in agents.items():
        mean, std = evaluate(agent)
        print(f"{name:10s} | Mean Reward: {mean:.2f} ± {std:.2f}")
