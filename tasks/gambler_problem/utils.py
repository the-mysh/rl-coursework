import numpy as np
import numpy.typing as npt


class GamblerProblemModel:
    def __init__(self, goal: int, success_probability: float):
        self.success_probability = success_probability
        self.goal = goal
        self.n_actions = goal // 2  # if goal is odd, max bet is less than half of the goal
        self.n_states = goal + 1
        self.n_nonterminal_states = goal - 1

        self.immediate_rewards = np.zeros(self.n_states)
        self.immediate_rewards[-1] = 1

        self.discount = 1

    def define_transition_probability_matrices(self) -> npt.NDArray[np.floating]:
        p_success = self.success_probability
        p_failure = 1 - p_success
        n_states = self.n_states

        transition_probs = np.zeros((self.n_actions, n_states, n_states))

        for action_idx in range(self.n_actions):
            action_value = action_idx + 1
            for state in range(n_states):
                if state - action_value < 0 or state + action_value >= n_states:
                    continue
                transition_probs[action_idx, state, state - action_value] = p_failure
                transition_probs[action_idx, state, state + action_value] = p_success
        return transition_probs