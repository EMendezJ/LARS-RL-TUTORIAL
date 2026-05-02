import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

## Normal run: python taxi_sarsa_qlearning.py
## Faster run: python taxi_sarsa_qlearning.py --episodes 1000 --eval-every 50 --eval-episodes 50


# ============================================================
# 1. Policy helpers
# ============================================================

def greedy(q_table, state):
    """
    Greedy policy: choose the action with the highest Q-value.
    """
    return int(np.argmax(q_table[state]))


def eps_greedy(q_table, state, epsilon=0.1):
    """
    Epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(q_table.shape[1])

    return greedy(q_table, state)


def evaluate_policy(env, q_table, num_episodes=100):
    """
    Evaluates the current greedy policy for several episodes.
    """
    episode_rewards = []

    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = greedy(q_table, state)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards))


# ============================================================
# 2. SARSA
# ============================================================

def train_sarsa(
    env,
    learning_rate=0.1,
    num_episodes=5000,
    epsilon=0.4,
    gamma=0.95,
    epsilon_decay=0.001,
    min_epsilon=0.01,
    eval_every=25,
    eval_episodes=100,
    model_dir="models",
):
    """
    Trains a SARSA agent on Taxi-v4.

    SARSA update:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

    SARSA is on-policy because the next action a' is selected using
    the same behavior policy used during training.
    """
    os.makedirs(model_dir, exist_ok=True)

    num_actions = env.action_space.n
    num_states = env.observation_space.n

    q_table = np.zeros((num_states, num_actions))

    eval_rewards = []
    eval_episodes_list = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

        action = eps_greedy(q_table, state, epsilon)

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action = eps_greedy(q_table, next_state, epsilon)

            td_target = reward + gamma * q_table[next_state][next_action] * (not done)
            td_error = td_target - q_table[state][action]

            q_table[state][action] += learning_rate * td_error

            state = next_state
            action = next_action

        if episode % eval_every == 0:
            mean_reward = evaluate_policy(
                env,
                q_table,
                num_episodes=eval_episodes,
            )

            eval_rewards.append(mean_reward)
            eval_episodes_list.append(episode)

        if episode == 500:
            np.save(os.path.join(model_dir, "Q_sarsa_500ep.npy"), q_table)

    np.save(os.path.join(model_dir, "Q_sarsa.npy"), q_table)

    return q_table, eval_rewards, eval_episodes_list


# ============================================================
# 3. Q-learning
# ============================================================

def train_q_learning(
    env,
    learning_rate=0.1,
    num_episodes=5000,
    epsilon=0.4,
    gamma=0.95,
    epsilon_decay=0.001,
    min_epsilon=0.01,
    eval_every=25,
    eval_episodes=100,
    model_dir="models",
):
    """
    Trains a Q-learning agent on Taxi-v4.

    Q-learning update:

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a Q(s',a) - Q(s,a)]

    Q-learning is off-policy because the update uses the best possible
    next action, even if that action was not actually selected.
    """
    os.makedirs(model_dir, exist_ok=True)

    num_actions = env.action_space.n
    num_states = env.observation_space.n

    q_table = np.zeros((num_states, num_actions))

    eval_rewards = []
    eval_episodes_list = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

        while not done:
            action = eps_greedy(q_table, state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            td_target = reward + gamma * np.max(q_table[next_state]) * (not done)
            td_error = td_target - q_table[state][action]

            q_table[state][action] += learning_rate * td_error

            state = next_state

        if episode % eval_every == 0:
            mean_reward = evaluate_policy(
                env,
                q_table,
                num_episodes=eval_episodes,
            )

            eval_rewards.append(mean_reward)
            eval_episodes_list.append(episode)

        if episode == 500:
            np.save(os.path.join(model_dir, "Q_qlearning_500ep.npy"), q_table)

    np.save(os.path.join(model_dir, "Q_qlearning.npy"), q_table)

    return q_table, eval_rewards, eval_episodes_list


# ============================================================
# 4. Visualization
# ============================================================

def moving_average(values, window=5):
    """
    Computes a simple moving average.
    """
    values = np.array(values, dtype=np.float32)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(
    sarsa_rewards,
    sarsa_episodes,
    qlearning_rewards,
    qlearning_episodes,
    window=5,
):
    """
    Plots smoothed evaluation rewards for SARSA and Q-learning.
    """
    sarsa_smoothed = moving_average(sarsa_rewards, window)
    qlearning_smoothed = moving_average(qlearning_rewards, window)

    sarsa_x = sarsa_episodes[: len(sarsa_smoothed)]
    qlearning_x = qlearning_episodes[: len(qlearning_smoothed)]

    plt.figure(figsize=(9, 5))

    plt.plot(
        sarsa_x,
        sarsa_smoothed,
        label="SARSA",
        linewidth=2,
    )

    plt.plot(
        qlearning_x,
        qlearning_smoothed,
        label="Q-learning",
        linewidth=2,
    )

    plt.title("Taxi-v4: SARSA vs Q-learning")
    plt.xlabel("Training episode")
    plt.ylabel("Mean evaluation reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def show_text_rollout(q_table, title, delay=0.5):
    """
    Shows one greedy policy rollout using Taxi-v4 ANSI rendering.
    """
    env = gym.make("Taxi-v4", render_mode="ansi")

    state, info = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    while not done:
        action = greedy(q_table, state)
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        state = next_state
        done = terminated or truncated
        step += 1

        print(env.render())
        print(f"Step: {step} | Action: {action} | Reward: {reward} | Total: {total_reward}")
        print("-" * 60)

        time.sleep(delay)

    print(f"{title} finished with total reward: {total_reward}")

    env.close()


# ============================================================
# 5. Main
# ============================================================

def set_seed(seed):
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-decay", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rollout", action="store_true")
    parser.add_argument("--rollout-delay", type=float, default=0.4)

    args = parser.parse_args()

    set_seed(args.seed)

    train_env = gym.make("Taxi-v4")

    print("Training SARSA...")
    q_sarsa, sarsa_rewards, sarsa_episodes = train_sarsa(
        train_env,
        learning_rate=args.learning_rate,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        model_dir=args.model_dir,
    )

    print("Training Q-learning...")
    q_qlearning, qlearning_rewards, qlearning_episodes = train_q_learning(
        train_env,
        learning_rate=args.learning_rate,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        model_dir=args.model_dir,
    )

    train_env.close()

    final_sarsa_score = sarsa_rewards[-1]
    final_qlearning_score = qlearning_rewards[-1]

    print(f"Final SARSA evaluation reward: {final_sarsa_score:.2f}")
    print(f"Final Q-learning evaluation reward: {final_qlearning_score:.2f}")

    print(f"SARSA Q-table saved to: {os.path.join(args.model_dir, 'Q_sarsa.npy')}")
    print(f"Q-learning Q-table saved to: {os.path.join(args.model_dir, 'Q_qlearning.npy')}")

    plot_training_curves(
        sarsa_rewards,
        sarsa_episodes,
        qlearning_rewards,
        qlearning_episodes,
        window=args.smooth_window,
    )

    if not args.no_rollout:
        show_text_rollout(
            q_sarsa,
            title="SARSA Greedy Policy Rollout",
            delay=args.rollout_delay,
        )

        show_text_rollout(
            q_qlearning,
            title="Q-learning Greedy Policy Rollout",
            delay=args.rollout_delay,
        )

    plt.show()


if __name__ == "__main__":
    main()