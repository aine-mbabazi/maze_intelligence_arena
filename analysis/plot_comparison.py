import matplotlib.pyplot as plt

agents = ["Random", "BFS", "Q-Learning"]
rewards = [-40, 3, 3]

plt.figure()
plt.bar(agents, rewards)
plt.ylabel("Average Reward")
plt.title("Agent Performance Comparison")
plt.show()
