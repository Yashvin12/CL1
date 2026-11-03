
# ==========================================
# REINFORCEMENT LEARNING USING Q-LEARNING
# MAZE ENVIRONMENT
# ==========================================

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Step 2: Create maze
maze = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

rows, cols = maze.shape

start = (0, 0)
goal = (4, 4)

# Step 3: Define actions
# 0 = Up, 1 = Down, 2 = Left, 3 = Right

actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

# Step 4: Initialize Q-table
q_table = np.zeros((rows * cols, 4))

# Learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

episodes = 5000
max_steps = 100

# Convert position into state number
def get_state(position):
    return position[0] * cols + position[1]

# Check whether movement is valid
def get_next_position(position, action):
    move = actions[action]

    new_row = position[0] + move[0]
    new_col = position[1] + move[1]

    if (0 <= new_row < rows and
        0 <= new_col < cols and
        maze[new_row, new_col] == 0):

        return (new_row, new_col)

    return position

# Step 5: Train the agent
for episode in range(episodes):

    position = start

    for step in range(max_steps):

        state = get_state(position)

        # Choose action using epsilon-greedy strategy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state])

        # Move agent
        next_position = get_next_position(position, action)

        # Calculate reward
        if next_position == goal:
            reward = 100
        elif next_position == position:
            reward = -5
        else:
            reward = -1

        next_state = get_state(next_position)

        # Q-learning update
        if next_position == goal:
            max_future_q = 0
        else:
            max_future_q = np.max(q_table[next_state])

        old_q = q_table[state, action]

        q_table[state, action] = old_q + learning_rate * (
            reward + discount_factor * max_future_q - old_q
        )

        position = next_position

        # Stop when goal is reached
        if position == goal:
            break

    # Reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training Completed!")
print("Total Episodes:", episodes)

# Step 6: Find optimal path
position = start
path = [start]

for step in range(max_steps):

    state = get_state(position)

    # Select best valid action
    valid_actions = []

    for action in range(4):
        next_pos = get_next_position(position, action)

        if next_pos != position:
            valid_actions.append(action)

    if not valid_actions:
        break

    action = max(
        valid_actions,
        key=lambda a: q_table[state, a]
    )

    position = get_next_position(position, action)
    path.append(position)

    if position == goal:
        break

print("\nLearned Optimal Path:")
print(path)

print("\nNumber of Steps:", len(path) - 1)

# Step 7: Display maze and learned path
plt.figure(figsize=(7, 7))

plt.imshow(maze, cmap="binary")

# Mark path
path_rows = [p[0] for p in path]
path_cols = [p[1] for p in path]

plt.plot(
    path_cols,
    path_rows,
    marker="o",
    color="red",
    linewidth=3,
    label="Learned Path"
)

# Mark start and goal
plt.scatter(
    start[1], start[0],
    color="green", s=200, label="Start"
)

plt.scatter(
    goal[1], goal[0],
    color="blue", s=200, label="Goal"
)

plt.xticks(range(cols))
plt.yticks(range(rows))
plt.title("Q-Learning: Maze Navigation")
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Display Q-table
print("\nFinal Q-Table:")
print(np.round(q_table, 2))
