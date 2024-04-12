import numpy as np


class QLearning:
    def __init__(self, states, actions, learning_rate=0.8, discount_factor=0.5, selection_strategy='epsilon-greedy',  epsilon=0.9):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.selection_strategy = selection_strategy
        self.epsilon = epsilon
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.qtable = self.initialize_qtable()

    def initialize_qtable(self):
        return np.zeros((self.n_states, self.n_actions))

    def get_action(self, state):
        state_index = self.states.index(state)
        if self.selection_strategy == 'epsilon-greedy':
            r = np.random.rand()
            if r > self.epsilon:
                action_index = np.random.randint(self.n_actions)
            else:
                action_index = np.argmax(self.qtable[state_index])
        if self.selection_strategy == 'boltzmann':
            max_Q = np.max(self.qtable)
            if max_Q == 0:
                action_index = np.random.randint(self.n_actions)
            else:
                denominator = np.sum(np.exp(self.qtable[state_index] / max_Q))
                probabilities = []
                for i, _ in enumerate(self.actions):
                    nominator = np.exp(self.qtable[state_index][i] / max_Q)
                    probabilities.append(nominator / denominator)
                action_index = np.random.choice(range(self.n_actions), p=probabilities)
        action = self.actions[action_index]
        return action

    def update_qtable(self, current_state, next_state, action, reward):
        current_state_id = self.states.index(current_state)
        next_state_id = self.states.index(next_state)
        action_id = self.actions.index(action)
        new_value = (1 - self.lr) * self.qtable[current_state_id][action_id]
        new_value = new_value + self.lr * (reward + self.gamma * (np.max(self.qtable[next_state_id])))
        self.qtable[current_state_id][action_id] = new_value

    def load_qtable(self, file_name):
        self.qtable = np.loadtxt(file_name)

    def save_qtable(self, file_name):
        np.savetxt(file_name, self.qtable)

if __name__ == '__main__':
    states = ['a', 'b', 'c', 'd', 'e']
    actions = ['up', 'down', 'right', 'left']
    ql = QLearning(states, actions)
    ql.load_qtable("qtable.txt")
    print(ql.qtable)

    action = ql.get_action('a')
    print(action)
    ql.update_qtable('a', 'b', action, 1)
    print(ql.qtable)
    ql.save_qtable("qtable.txt")
