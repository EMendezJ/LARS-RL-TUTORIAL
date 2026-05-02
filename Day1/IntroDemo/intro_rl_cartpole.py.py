import argparse
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# run the script: python intro_rl_cartpole.py

def collect_random_policy_rewards(env_name="CartPole-v1", episodes=10):
    """
    Runs a random policy for a fixed number of episodes.
    Returns the cumulative reward from each episode.
    """
    env = gym.make(env_name)

    episode_rewards = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    env.close()

    return episode_rewards


def animate_random_policy(env_name="CartPole-v1", max_steps=200, delay=0.03):
    """
    Runs one random-policy episode and animates it using matplotlib.
    Works as a normal Python script.
    """
    env = gym.make(env_name, render_mode="rgb_array")

    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    fig, ax = plt.subplots(figsize=(7, 5))

    while not done and step < max_steps:
        frame = env.render()

        ax.clear()
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(
            f"CartPole Random Policy | Step: {step} | Reward: {total_reward:.0f}"
        )

        plt.pause(delay)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated
        step += 1

    frame = env.render()

    ax.clear()
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title(
        f"Episode Finished | Steps: {step} | Total Reward: {total_reward:.0f}"
    )

    plt.pause(1.0)
    env.close()


def plot_episode_rewards(rewards):
    """
    Plots cumulative reward per episode.
    """
    episodes = np.arange(1, len(rewards) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, marker="o", label="Episode reward")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("CartPole Random Policy Performance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.03)
    parser.add_argument("--no-animation", action="store_true")

    args = parser.parse_args()

    rewards = collect_random_policy_rewards(
        env_name="CartPole-v1",
        episodes=args.episodes,
    )

    plot_episode_rewards(rewards)

    if not args.no_animation:
        animate_random_policy(
            env_name="CartPole-v1",
            max_steps=args.max_steps,
            delay=args.delay,
        )

    plt.show()


if __name__ == "__main__":
    main()