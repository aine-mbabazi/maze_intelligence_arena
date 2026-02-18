import random


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        return random.choice(self.env.actions)
