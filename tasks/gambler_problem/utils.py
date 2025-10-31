import numpy as np
import numpy.typing as npt


class GamblerPolicy:
    def __init__(self, goal: int, success_probability: float):
        self.success_probability = success_probability
        self.goal = goal
        self.n_states = goal + 1
        self.n_nonterminal_states = goal - 1

        self.policy_estimate = np.ones(self.n_nonterminal_states, dtype=int)

        self.immediate_rewards = np.zeros(self.n_states)
        self.immediate_rewards[-1] = 1

    def define_ptpm(self) -> npt.NDArray[np.floating]:

        # policy transition probability matrix: s -> s'
        ptpm = np.zeros((self.n_states, self.n_states))
        for idx, action in enumerate(self.policy_estimate):
            assert isinstance(action, np.integer)
            starting_capital = idx + 1  # policy defined only for non-terminal states; idx 0 is for state of 1$
            success_capital = starting_capital + action  # action is number of dollars to bet
            failure_capital = starting_capital - action

            ptpm[starting_capital, success_capital] = self.success_probability
            ptpm[starting_capital, failure_capital] = 1 - self.success_probability
        return ptpm
