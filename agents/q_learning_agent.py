import numpy as np
import random




class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((env.size, env.size, 4))

    def act(self, state):
        x, y = state
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        best_next = np.max(self.q_table[nx, ny])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[x, y, action]

        self.q_table[x, y, action] += self.alpha * td_error

    def train(self, episodes=500, max_steps=100):
        for _ in range(episodes):
            state = self.env.reset()
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                if done:
                    break

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)