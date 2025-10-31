import numpy as np
import numpy.typing as npt


class GamblerPolicy:
    def __init__(self, goal: int, success_probability: float):
        self.success_probability = success_probability
        self.goal = goal
        self.n_states = goal + 1
        self.n_nonterminal_states = goal - 1

        self.policy_estimate = np.ones(self.n_nonterminal_states, dtype=int)

    def define_ptpm(self) -> npt.NDArray[np.floating]:

        # policy transition probability matrix: s -> s'
        ptpm = np.zeros((self.n_nonterminal_states, self.n_states))
        for idx, action in enumerate(self.policy_estimate):
            assert isinstance(action, np.integer)
            starting_capital = idx + 1  # state of 1$ is at index 0
            success_capital = starting_capital + action  # action is number of dollars to bet
            failure_capital = starting_capital - action

            # subtract ones when indexing on starting state axis only; the other axis starts with terminal state 0
            ptpm[idx, success_capital] = self.success_probability
            ptpm[idx, failure_capital] = 1 - self.success_probability
        return ptpm
