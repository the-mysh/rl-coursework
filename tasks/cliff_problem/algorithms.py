import numpy as np
import numpy.typing as npt
from random import random, choice
from typing import NamedTuple
from abc import abstractmethod, ABC

from cliff_game import CliffGame, Action
from plotting import plot_q_values, plot_policy


class State(NamedTuple):
    x: int
    y: int


class Algorithm(ABC):
    _name: str = NotImplemented

    def __init__(self, game: CliffGame, alpha=0.5, gamma=1., epsilon=0.15):
        self.game = game
        self.actions = list(Action.__members__.values())
        self.q_values = np.zeros((*self.game.scene.shape, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_idx_for_state_and_action(self, state: State, action: Action):
        return *state, self.actions.index(action)

    def choose_action(self, state: State, explore: bool = True) -> Action:
        if explore and random() < self.epsilon:
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

    @abstractmethod
    def run_episode(self, dry: bool = False, max_steps: int = 1000, keep_trajectory: bool = False
                    ) -> tuple[list[State], list[Action], int]:
        pass

    def run(self, n_episodes: int, **kwargs):
        rewards = []

        for i in range(n_episodes):
            self.game.reset()
            _, _, r = self.run_episode(**kwargs)
            rewards.append(r)

        return rewards

    def run_multiple(self, n_episodes: int, n_runs: int, **kwargs):
        combined_rewards = None
        combined_q_values = None

        for i in range(n_runs):
            self.reset()
            rewards = self.run(n_episodes, **kwargs)

            combined_rewards = self.average_runs(combined_rewards, np.array(rewards), i+1)
            combined_q_values = self.average_runs(combined_q_values, self.q_values[:], i+1)

        self.q_values = combined_q_values
        return combined_rewards

    @staticmethod
    def average_runs(old: npt.NDArray | None, new: npt.NDArray, idx: int):
        if old is None:
            return new

        return old + (new - old) / idx

    def get_current_policy(self) -> tuple[npt.NDArray[np.integer], npt.NDArray[Action], npt.NDArray[np.bool_]]:
        policy_idx = np.argmax(self.q_values, axis=-1)
        terminal = ~np.any(self.q_values, axis=-1)

        policy_actions = np.array(self.actions)[policy_idx]
        policy_actions[terminal] = ""

        return policy_idx, policy_actions, terminal

    def plot_q_values(self):
        plot_q_values(self.q_values, self.actions, title=f"{self._name} Q-values")

    def plot_current_policy(self, trajectory: list[State] | None = None, max_steps: int = 1000, **kwargs):
        x_components = []
        y_components = []

        for action in self.actions:
            y_change, x_change = self.game.movement[action]
            x_components.append(x_change)
            y_components.append(y_change)

        policy_idx, _, terminal = self.get_current_policy()

        u = np.array(x_components).astype(float)[policy_idx]
        v = np.array(y_components).astype(float)[policy_idx]
        v = -v  # y-axis is inverted; going up in numbers is going down in the image

        u[terminal] = np.nan
        v[terminal] = np.nan

        if trajectory is None:
            self.game.reset()
            trajectory, _, _ = self.run_episode(dry=True, keep_trajectory=True, max_steps=max_steps)

        plot_policy(u, v, trajectory, title=f"{self._name} computed policy", **kwargs)


class Sarsa(Algorithm):
    _name = 'SARSA'

    def run_episode(self, dry: bool = False, max_steps=1000, keep_trajectory: bool = False
                    ) -> tuple[list[State], list[Action], int]:
        state = State(*self.game.agent_pos)
        action: Action = self.choose_action(state, explore=not dry)
        states_sequence = [state]
        actions_sequence = []

        game_over = False
        total_reward = 0
        steps = 0
        while not game_over:
            if steps > max_steps:
                break

            if keep_trajectory:
                actions_sequence.append(action)

            reward, game_over = self.take_action(action)
            total_reward += reward
            new_state = State(*self.game.agent_pos)
            new_action = self.choose_action(new_state, explore=not dry)

            if not dry:
                self.update_q_values(state, action, reward, new_state, new_action)

            state = new_state
            action = new_action
            if keep_trajectory:
                states_sequence.append(state)
            steps += 1

        return states_sequence, actions_sequence, total_reward

    def update_q_values(self, state: State, action: Action, reward: int, new_state: State, new_action: Action):
        idx = self.get_q_idx_for_state_and_action(state, action)

        update = self.alpha * (self.compute_target(reward, new_state, new_action) - self.q_values[idx])
        self.q_values[idx] += update

    def compute_target(self, reward: int, new_state: State, new_action: Action) -> float:
        idx = self.get_q_idx_for_state_and_action(new_state, new_action)
        return reward + self.gamma * self.q_values[idx]


class QLearning(Algorithm):
    _name = 'Q-Learning'

    def run_episode(self, dry: bool = False, max_steps=1000, keep_trajectory: bool = False
                    ) -> tuple[list[State], list[Action], int]:

        state = State(*self.game.agent_pos)
        states_sequence = [state]
        actions_sequence = []

        game_over = False
        total_reward = 0
        steps = 0
        while not game_over:
            if steps > max_steps:
                break

            action = self.choose_action(state, explore=not dry)

            if keep_trajectory:
                actions_sequence.append(action)

            reward, game_over = self.take_action(action)
            total_reward += reward
            new_state = State(*self.game.agent_pos)

            if not dry:
                self.update_q_values(state, action, reward, new_state)

            state = new_state
            if keep_trajectory:
                states_sequence.append(state)
            steps += 1

        return states_sequence, actions_sequence, total_reward

    def update_q_values(self, state: State, action: Action, reward: int, new_state: State):
        idx = self.get_q_idx_for_state_and_action(state, action)

        update = self.alpha * (self.compute_target(reward, new_state) - self.q_values[idx])
        self.q_values[idx] += update

    def compute_target(self, reward: int, new_state: State) -> float:

        best_q = self.q_values[*new_state].max()
        return reward + self.gamma * best_q
