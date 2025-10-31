import numpy as np
import numpy.typing as npt


class GamblerProblemModel:
    def __init__(self, goal: int, success_probability: float):
        self.success_probability = success_probability
        self.goal = goal
        self.n_states = goal + 1
        self.n_nonterminal_states = goal - 1

        self.immediate_rewards = np.zeros(self.n_states)
        self.immediate_rewards[-1] = 1

    def propose_policy_estimate(self) -> npt.NDArray[np.integer]:
        return np.ones(self.n_nonterminal_states, dtype=int)

    def define_ptpm(self, policy_estimate) -> npt.NDArray[np.floating]:

        self._check_policy(policy_estimate)

        # policy transition probability matrix: s -> s'
        ptpm = np.zeros((self.n_states, self.n_states))
        for idx, action in enumerate(policy_estimate):
            assert isinstance(action, np.integer)
            starting_capital = idx + 1  # policy defined only for non-terminal states; idx 0 is for state of 1$
            success_capital = starting_capital + action  # action is number of dollars to bet
            failure_capital = starting_capital - action

            ptpm[starting_capital, success_capital] = self.success_probability
            ptpm[starting_capital, failure_capital] = 1 - self.success_probability
        return ptpm

    def _check_policy(self, policy_estimate: npt.NDArray[np.integer]):
        if not isinstance(policy_estimate, np.ndarray):
            raise TypeError(f"Expected a numpy array, got type {type(policy_estimate)}: {policy_estimate}")

        if len(policy_estimate) != self.n_nonterminal_states:
            raise ValueError(f"Policy estimate should have {self.n_nonterminal_states} entries; "
                             f"got {len(policy_estimate)}")

        if not np.issubdtype(policy_estimate.dtype, np.integer):
            raise TypeError(f"Expected an integer-valued array, got {policy_estimate.dtype}")