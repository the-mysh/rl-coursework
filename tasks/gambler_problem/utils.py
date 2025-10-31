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

        self.discount = 1

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

    def evaluate_policy(
            self,
            policy: npt.NDArray[np.integer],
            n_steps: int,
            initial_state_values: npt.NDArray[np.floating] | None = None,
            keep_track: bool = False
    ) -> npt.NDArray[np.floating]:

        if initial_state_values is None:
            state_values = np.zeros(self.n_states)
        else:
            if initial_state_values.shape != self.n_states:
                raise ValueError(f"Initial values array must have shape {(self.n_states,)}")
            state_values = initial_state_values

        if keep_track:
            track = np.empty((n_steps+1, self.n_states), dtype=state_values.dtype)
            track[0] = state_values
        else:
            track = None

        r = self.immediate_rewards
        ptpm = self.define_ptpm(policy)
        d = self.discount

        for i in range(10):
            new_values_of_states = r + d * np.matmul(ptpm, state_values)
            new_values_of_states = np.round(new_values_of_states, 4)
            if keep_track:
                track[i+1] = new_values_of_states
            state_values = new_values_of_states

        if keep_track:
            return track
        return state_values
