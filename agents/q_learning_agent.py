import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros(
            (env.state_size, env.action_size)
        )


    def act(self, state):
        x, y = state

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        # Exploitation
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        # max Q(s', ·)
        best_next = np.max(self.q_table[nx, ny])

        # r + γ max Q(s', ·)
        td_target = reward + self.gamma * best_next

        # [ r + γ max Q(s', ·) − Q(s,a) ]
        td_error = td_target - self.q_table[x, y, action]

        # Q(s,a) ← Q(s,a) + α [ ... ]
        self.q_table[x, y, action] += self.alpha * td_error

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
         self.q_table = np.load(path)
