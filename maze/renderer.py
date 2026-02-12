def render(maze):
    size = maze.size
    agent_x, agent_y = maze.agent_pos
    goal_x, goal_y = maze.goal

    print("\nMaze:")
    for i in range(size):
        row = ""
        for j in range(size):
            if (i, j) == (agent_x, agent_y):
                row += " A "
            elif (i, j) == (goal_x, goal_y):
                row += " G "
            elif maze.grid[i, j] == 1:
                row += " # "
            else:
                row += " . "
        print(row)
    print()
