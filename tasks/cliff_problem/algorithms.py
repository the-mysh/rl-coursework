import numpy as np
from random import random, choice
from typing import NamedTuple

from cliff_game import CliffGame, Action


class State(NamedTuple):
    x: int
    y: int


class Sarsa:
    def __init__(self, game: CliffGame, alpha=0.5, gamma=1., epsilon=0.15):
        self.game = game
        self.actions = list(Action.__members__.values())
        self.q_values = np.zeros((*self.game.scene.shape, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_idx_for_state_and_action(self, state: State, action: Action):
        return *state, self.actions.index(action)

    def update_q_values(self, state: State, action: Action, reward: int, new_state: State, new_action: Action):
        idx = self.get_q_idx_for_state_and_action(state, action)

        update = self.alpha * (self.compute_target(reward, new_state, new_action) - self.q_values[idx])
        self.q_values[idx] += update

    def compute_target(self, reward: int, new_state: State, new_action: Action) -> float:
        idx = self.get_q_idx_for_state_and_action(new_state, new_action)
        return reward + self.gamma * self.q_values[idx]

    def choose_action(self, state: State) -> Action:
        if random() < self.epsilon:
            return choice(self.actions)

        action_q_values = self.q_values[*state, :]
        best_action_indices = np.where(action_q_values == np.max(action_q_values))[0]
        if len(best_action_indices) == 1:
            best_action_index = best_action_indices[0]
        else:
            best_action_index = choice(best_action_indices)

        return self.actions[best_action_index]

    def take_action(self, action: Action) -> tuple[int, bool]:
        return self.game.move(action)

    def reset(self):
        self.q_values = np.zeros((*self.game.scene.shape, len(self.actions)))
        self.game.reset()

    def run(self):
        state = State(*self.game.agent_pos)
        action: Action = self.choose_action(state)
        states_sequence = [state]
        actions_sequence = []

        game_over = False
        total_reward = 0
        while not game_over:
            print('+', end='')
            actions_sequence.append(action)

            reward, game_over = self.take_action(action)
            total_reward += reward
            new_state = State(*self.game.agent_pos)
            new_action = self.choose_action(new_state)

            self.update_q_values(state, action, reward, new_state, new_action)

            state = new_state
            action = new_action
            states_sequence.append(state)

        return states_sequence, actions_sequence, total_reward

