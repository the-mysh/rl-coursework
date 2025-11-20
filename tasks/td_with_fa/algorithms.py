from abc import ABC, abstractmethod
from random import random
import numpy as np
import numpy.typing as npt

from problem import Problem



class Algorithm(ABC):
    def __init__(self, problem: Problem, n_params: int, alpha=0.01, gamma=0.99, reward=0):
        self.problem = problem
        self.phi = self.define_weight_matrices(problem.N_STATES, n_params)

        self.alpha = alpha
        self.gamma = gamma
        self.reward = reward

    @staticmethod
    def define_weight_matrices(n_states: int, n_params: int):
        phi_a = np.zeros((n_states, n_params), dtype=int)
        phi_a[:, n_params//2] = 1
        phi_a[np.arange(n_states-1), np.arange(n_states-1)] = 2
        phi_a[n_states-1, n_states-1] = 1
        phi_a[n_states-1, n_states] = 2

        phi_b = np.zeros((n_states, n_params), dtype=int)
        phi_b[np.arange(n_states), n_params//2 + n_params%2 + np.arange(n_states)] = 1

        return np.stack([phi_a, phi_b]).swapaxes(0, 1)

    @classmethod
    def choose_action(cls):
        if 7 * random() < 1:
            return Problem.Action.A
        else:
            return Problem.Action.B

    @abstractmethod
    def run(self, w_init, n_steps: int):
        pass


class Sarsa(Algorithm):
    def run(self, w_init: npt.NDArray[np.floating], n_steps: int, keep_trajectory: bool = False
            ) -> tuple[npt.NDArray[np.floating], list[Problem.State], list[Problem.Action]]:
        w = w_init[:]
        state = np.random.choice(self.problem.STATES)
        action = self.choose_action()
        states_sequence = [state]
        actions_sequence = []
        w_norms = [np.linalg.norm(w)]

        for _ in range(20):
            if keep_trajectory:
                actions_sequence.append(action)

            new_state = self.problem.take_action(state, action)
            new_action = self.choose_action()

            phi_st = self.phi[state.value, action.value]
            phi_new_st = self.phi[new_state.value, new_action.value]
            estimate = phi_st @ w
            target = self.reward + self.gamma * phi_new_st @ w
            w += self.alpha * phi_st * (target - estimate)

            state = new_state
            action = new_action
            if keep_trajectory:
                states_sequence.append(state)
            w_norms.append(np.linalg.norm(w))

        return np.array(w_norms), states_sequence, actions_sequence
