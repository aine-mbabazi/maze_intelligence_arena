import numpy as np

class Maze:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos

        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        dx, dy = moves[action]
        nx, ny = x + dx, y + dy

        # Check bounds and walls
        if (
            0 <= nx < self.size
            and 0 <= ny < self.size
            and self.grid[nx, ny] == 0
        ):
            self.agent_pos = (nx, ny)

        reward = -1
        done = self.agent_pos == self.goal
        if done:
            reward = 10

        return self.agent_pos, reward, done
