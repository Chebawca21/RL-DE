import gymnasium as gym
import numpy as np
from qlearning import QLearning
import time


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    actions = range(env.action_space.n)
    states = range(env.observation_space.n)

    ql = QLearning(states, actions, epsilon=0.0)

    N = 2000
    for _ in range(N):
        state, info = env.reset()
        action = ql.get_action(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        ql.epsilon = ql.epsilon + (0.9 / N)
        while not terminated:
            ql.update_qtable(state, new_state, action, reward)
            state = new_state
            action = ql.get_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
        ql.update_qtable(state, new_state, action, reward)
    env.close()

    env = gym.make('Taxi-v3', render_mode="human")
    for _ in range(5):
        state, info = env.reset()
        action = ql.get_action(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        while not terminated:
            env.render()
            # ql.update_qtable(state, new_state, action, reward)
            state = new_state
            action = ql.get_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
    env.close()
