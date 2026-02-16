import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curve(csv_path="data/q_learning_rewards.csv"):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["episode"], df["total_reward"])
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning: Reward per Episode")
    plt.show()


if __name__ == "__main__":
    plot_learning_curve()
