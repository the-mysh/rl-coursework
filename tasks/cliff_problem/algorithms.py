import numpy as np
import numpy.typing as npt
from random import random, choice
from typing import NamedTuple
import matplotlib.pyplot as plt

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

    def run(self, verbose: bool = False, dry: bool = False, max_steps=1000) -> tuple[list[State], list[Action], int]:
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
            if verbose:
                print('+', end='')
            actions_sequence.append(action)

            reward, game_over = self.take_action(action)
            total_reward += reward
            new_state = State(*self.game.agent_pos)
            new_action = self.choose_action(new_state, explore=not dry)

            if not dry:
                self.update_q_values(state, action, reward, new_state, new_action)

            state = new_state
            action = new_action
            states_sequence.append(state)
            steps += 1

        return states_sequence, actions_sequence, total_reward

    def get_current_policy(self) -> tuple[npt.NDArray[np.integer], npt.NDArray[Action], npt.NDArray[np.bool_]]:
        policy_idx = np.argmax(self.q_values, axis=-1)
        terminal = ~np.any(self.q_values, axis=-1)

        policy_actions = np.array(self.actions)[policy_idx]
        policy_actions[terminal] = ""

        return policy_idx, policy_actions, terminal

    def plot_q_values(self):
        vmin = self.q_values.min()
        vmax = self.q_values.max()

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for i, action in enumerate(self.actions):
            ax = axes[i // 2, i % 2]
            im = ax.imshow(self.q_values[:, :, i], vmin=vmin, vmax=vmax, cmap='viridis')
            ax.set_title(action.name)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    def plot_current_policy(self, trajectory: list[State] | None = None, color='navy'):
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
            trajectory, _, _ = self.run(dry=True)
        y_coords, x_coords = list(zip(*trajectory))

        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig, axes = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(10, 6))

            for ax in axes:
                ax.yaxis.set_inverted(True)
                ax.set_aspect('equal')
                ax.set_xticks(np.arange(policy_idx.shape[1] + 1) - 0.5)
                ax.set_yticks(np.arange(policy_idx.shape[0]) - 0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlim(-0.5, policy_idx.shape[1] - 0.5)
                ax.set_ylim(policy_idx.shape[0] - 0.5, -0.5)
                ax.tick_params(axis='both', which='major', length=0)

            axes[0].quiver(u, v, pivot='mid', color=color)
            axes[0].set_title("Policy")

            axes[1].plot(x_coords, y_coords, marker='o', lw=0.5, color=color, alpha=0.5)

            mkw = dict(lw=0, markerfacecolor='none', markeredgecolor=color, markersize=10, zorder=0)
            axes[1].plot([x_coords[0]], [y_coords[0]], marker='D', **mkw, label="Start")
            axes[1].plot([x_coords[-1]], [y_coords[-1]], marker='s', **mkw, label="Finish")
            axes[1].legend(frameon=True, framealpha=0.5, fancybox=True)

            axes[1].set_title("Greedy trajectory")

            fig.suptitle("SARSA results")

