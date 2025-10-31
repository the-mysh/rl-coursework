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

    @staticmethod
    def round_up(arr, precision):
        factor = 10**precision
        return np.round(np.ceil(arr * factor) / factor, precision)

    def run_value_iteration(self, convergence: float = 10e-4, max_iter: int = 1000, keep_track: bool = False):
        v_track = []
        pi_track = []
        err_track = []

        transition_probs = self.define_transition_probability_matrices()
        imr = self.immediate_rewards

        v = np.zeros(self.n_states)  # initial value 'function'

        for i in range(max_iter):
            comp = np.matvec(transition_probs, v + imr)
            comp = self.round_up(comp, 4)
            new_v = np.max(comp, axis=0)
            new_pi = np.argmax(comp, axis=0)
            err = np.max(np.abs(v - new_v))

            if keep_track:
                v_track.append(new_v)
                pi_track.append(new_pi)
                err_track.append(err)

            if err < convergence:
                break

            v = new_v

        sl = slice(1, -1)
        if keep_track:
            v_track_arr = np.stack(v_track)[:, sl]
            pi_track_arr = np.stack(pi_track)[:, sl]
            err_track_arr = np.array(err_track)
            return v_track_arr, pi_track_arr, err_track_arr

        return new_v[sl], new_pi[sl], err
