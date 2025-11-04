import numpy as np
import numpy.typing as npt
from enum import Enum, auto
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class VIApproach(Enum):
    VECTORIZED = auto()
    IN_PLACE = auto()


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

    def _sweep_vectorised(self, v, transition_probs, precision: int = 4):
        comp = np.matvec(transition_probs, self.discount * v + self.immediate_rewards)
        comp = comp
        new_v = np.max(comp, axis=0)
        new_pi = np.argmax(np.round(comp, precision), axis=0) + 1  # bet value is best action index + 1 (0-based indexing; min bet is 1)
        err = np.max(np.abs(v - new_v))
        return new_v, new_pi, err

    def _sweep_in_place(self, v, transition_probs, precision: int = 4):
        discount = self.discount
        imr = self.immediate_rewards

        v = v.copy()

        pi = np.zeros(self.n_states, dtype=int)
        max_err = 0

        for state in range(1, self.n_states-1):
            old_state_value = v[state]

            best_action = None
            best_q_value = -np.inf
            for action_idx in range(min(self.n_actions, self.goal-state)):
                p = transition_probs[action_idx, state]  # vector: (<n_states>,)
                possible_q_value = p @ (imr + discount * v)

                # round to specified precision only for extracting policy (determining best action for this state)
                if (best_action is None) or (np.round(possible_q_value, precision) > np.round(best_q_value, precision)):
                    best_action = action_idx + 1  # min action index is 0, corresponds to bet = 1

                if possible_q_value > best_q_value:
                    best_q_value = possible_q_value

            pi[state] = best_action
            v[state] = best_q_value
            max_err = max(max_err, np.abs(best_q_value-old_state_value))

        return v, pi, max_err

    def run_value_iteration(self, convergence: float = 10e-4, max_iter: int = 1000, keep_track: bool = False,
                            approach: VIApproach = VIApproach.VECTORIZED, precision: int = 4):
        v_track = []
        pi_track = []
        err_track = []

        transition_probs = self.define_transition_probability_matrices()

        match approach:
            case VIApproach.VECTORIZED:
                sweep_func = self._sweep_vectorised
            case VIApproach.IN_PLACE:
                sweep_func = self._sweep_in_place
            case _:
                raise ValueError(f'Invalid approach specified: {approach}')

        v = np.zeros(self.n_states)  # initial value 'function'

        for i in range(max_iter):
            new_v, new_pi, err = sweep_func(v, transition_probs, precision=precision)

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


def plot_value_iteration(vs, pis):
    x = np.arange(pis.shape[1]) + 1
    kw = dict(ls='--', marker='.', lw=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax in axes:
        ax.grid(color='lightgrey')
        ax.set_xlabel("State (capital)")

    for idx in (0, 1, 2, -1):
        axes[0].plot(
            x,
            vs[idx],
            label=(f"{len(vs)} (final)" if idx==-1 else f"{idx+1}"),
            alpha=0.7,
            **kw
        )

    axes[0].legend(title="Iteration", fancybox=True, framealpha=0.5)
    axes[0].set_title("Value function - iteration results")
    axes[0].set_ylabel("State value")

    axes[1].plot(x, pis[-1], **kw)
    axes[1].yaxis.set_major_locator(MultipleLocator(5))
    axes[1].set_title("Optimal policy")
    axes[1].set_ylabel("Policy (bet value)")

    fig.subplots_adjust(hspace=0.3)


def plot_estimate_evolution(estimates, label: str, log_scale_diff: bool = False, cbar_shrink=0.8, fig_height=7, equalise_diff: bool = False):
    fig, axes = plt.subplots(3, 1, figsize=(10, fig_height), layout='constrained', gridspec_kw={'height_ratios': [3, 3, 1]})
    n_estimates, n_states = estimates.shape

    im0 = axes[0].imshow(estimates, cmap='jet', extent=(0.5, n_states+0.5, n_estimates+0.5, 0.5))
    fig.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.04)
    axes[0].set_xlabel("State (capital, $)")
    axes[0].set_ylabel("Iteration")
    axes[0].set_title(label)

    diff = np.diff(estimates, axis=0)
    if equalise_diff:
        m = np.max(np.abs(diff)) // 2
        vmin, vmax = -m, m
    else:
        vmin, vmax = None, None
    im1 = axes[1].imshow(diff, cmap='coolwarm', vmin=vmin, vmax=vmax, extent=(0.5, n_states+0.5, n_estimates+0.5, 1.5))
    fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.04)
    axes[1].set_xlabel("State (capital, $)")
    axes[1].set_ylabel("Iteration")
    axes[1].set_title(f"Difference over iterations")

    max_diff = np.max(diff, axis=1)
    x = np.arange(len(max_diff)) + 2
    axes[2].plot(x, max_diff, lw=1, marker='.')
    axes[2].grid(color='lightgrey')

    if log_scale_diff:
        axes[2].set_yscale('log')

    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Max error" + ("\n(log scale)" if log_scale_diff else ""))
    axes[2].set_title(f"Maximum difference per iteration")


