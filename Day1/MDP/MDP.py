import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ACTION_SYMBOLS = ["←", "↓", "→", "↑"]  # 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP

# Run Demo: python frozen_lake_policy.py


def value_iteration(env, gamma=0.99, max_iterations=1_000, threshold=1e-10):
    """
    Performs value iteration for a discrete Gymnasium environment.
    Returns the optimal value function.
    """
    assert hasattr(env, "P"), "This function requires an environment with a P transition table."

    value_table = np.zeros(env.observation_space.n)

    for _ in range(max_iterations):
        old_value_table = value_table.copy()

        for state in range(env.observation_space.n):
            q_values = []

            for action in range(env.action_space.n):
                q_sa = 0.0

                for prob, next_state, reward, terminated in env.P[state][action]:
                    future_value = 0.0 if terminated else old_value_table[next_state]
                    q_sa += prob * (reward + gamma * future_value)

                q_values.append(q_sa)

            value_table[state] = max(q_values)

        if np.sum(np.abs(old_value_table - value_table)) <= threshold:
            break

    return value_table


def extract_policy(env, value_table, gamma=0.99):
    """
    Extracts the greedy policy from a value function.
    """
    assert hasattr(env, "P"), "This function requires an environment with a P transition table."

    policy = np.zeros(env.observation_space.n, dtype=int)

    for state in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            for prob, next_state, reward, terminated in env.P[state][action]:
                future_value = 0.0 if terminated else value_table[next_state]
                q_values[action] += prob * (reward + gamma * future_value)

        policy[state] = np.argmax(q_values)

    return policy


def plot_policy(env, policy, title="Optimal Policy"):
    """
    Plots the final learned policy on top of the FrozenLake map.
    """
    grid_size = int(np.sqrt(env.observation_space.n))

    color_map = {
        b"S": "#ADD8E6",  # Start
        b"F": "#FFFFFF",  # Frozen
        b"H": "#808080",  # Hole
        b"G": "#90EE90",  # Goal
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(True)

    for state in range(env.observation_space.n):
        row, col = divmod(state, grid_size)
        tile = env.desc[row][col]

        ax.add_patch(
            plt.Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                color=color_map[tile],
                ec="black",
            )
        )

        if tile == b"H":
            text = "H"
        elif tile == b"G":
            text = "G"
        elif tile == b"S":
            text = "S"
        else:
            text = ACTION_SYMBOLS[policy[state]]

        ax.text(
            col,
            row,
            text,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show(block=False)


def simulate_policy(env, policy, episodes=1, delay=0.6):
    """
    Simulates the learned policy in a normal Python script using matplotlib.
    No Jupyter/IPython display calls are needed.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        total_reward = 0.0

        while not (terminated or truncated):
            frame = env.render()

            ax.clear()
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(
                f"Episode {episode + 1} | Step {step} | "
                f"State {obs} | Action {ACTION_SYMBOLS[policy[obs]]}"
            )

            plt.pause(delay)

            action = int(policy[obs])
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            step += 1

        frame = env.render()

        ax.clear()
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(
            f"Episode {episode + 1} finished | "
            f"Steps: {step} | Reward: {total_reward}"
        )

        plt.pause(1.0)

    plt.show()


def main():
    gamma = 0.99

    # Environment for planning.
    # We unwrap this one to access env.P.
    planning_env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        disable_env_checker=True,
    ).unwrapped

    optimal_value_function = value_iteration(
        planning_env,
        gamma=gamma,
    )

    optimal_policy = extract_policy(
        planning_env,
        optimal_value_function,
        gamma=gamma,
    )

    plot_policy(
        planning_env,
        optimal_policy,
        title="Optimal Policy for FrozenLake",
    )

    # Separate environment for visualization.
    # This one needs render_mode="rgb_array".
    render_env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode="rgb_array",
        disable_env_checker=True,
    )

    simulate_policy(
        render_env,
        optimal_policy,
        episodes=1,
        delay=0.6,
    )

    render_env.close()


if __name__ == "__main__":
    main()