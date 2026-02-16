import csv
import os


def save_rewards(rewards, filename="data/q_learning_rewards.csv"):
    os.makedirs("data", exist_ok=True)

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward"])

        for i, reward in enumerate(rewards):
            writer.writerow([i, reward])
