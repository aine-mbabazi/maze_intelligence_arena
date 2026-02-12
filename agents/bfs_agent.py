from collections import deque

class BFSAgent:
    def __init__(self, maze):
        self.maze = maze
        self.path = self._bfs()

    def _bfs(self):
        start = self.maze.start
        goal = self.maze.goal
        size = self.maze.size
        grid = self.maze.grid

        queue = deque([start])
        came_from = {start: None}

        moves = {
            (-1, 0): 0,  # up
            (1, 0): 1,   # down
            (0, -1): 2,  # left
            (0, 1): 3,   # right
        }

        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for (dx, dy), action in moves.items():
                nx, ny = current[0] + dx, current[1] + dy
                next_pos = (nx, ny)

                if (
                    0 <= nx < size
                    and 0 <= ny < size
                    and grid[nx, ny] == 0
                    and next_pos not in came_from
                ):
                    queue.append(next_pos)
                    came_from[next_pos] = (current, action)

        # Reconstruct path
        path = []
        current = goal
        while came_from[current] is not None:
            prev, action = came_from[current]
            path.append(action)
            current = prev

        path.reverse()
        return path

    def act(self):
        if self.path:
            return self.path.pop(0)
        return 0 
