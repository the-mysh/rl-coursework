from enum import Enum
import numpy as np


class Action(Enum):
    A = 0
    B = 1


class Problem:
    class State(Enum):
        S1 = 0
        S2 = 1
        S3 = 2
        S4 = 3
        S5 = 4
        S6 = 5
        S7 = 6

    STATES = list(State.__members__.values())
    N_STATES = len(STATES)

    def __init__(self):
        self._transition_probs = self.define_transition_probabilities()

    @property
    def transition_probs(self):
        return self._transition_probs

    @property
    def transition_probs_a(self):
        return self._transition_probs[Action.A.value]

    @property
    def transition_probs_b(self):
        return self._transition_probs[Action.B.value]

    @classmethod
    def define_transition_probabilities(cls):
        tp_a = np.zeros((cls.N_STATES, cls.N_STATES))
        tp_a[:, -1] = 1

        tp_b = np.full((cls.N_STATES, cls.N_STATES), 1/6)
        tp_b[:, -1] = 0

        return np.stack([tp_a, tp_b])

    def take_action(self, state: State, action: Action) -> State:
        target_probs = self.transition_probs[action.value, state.value]
        return np.random.choice(self.STATES, p=target_probs)

