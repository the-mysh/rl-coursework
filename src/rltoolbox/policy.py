from abc import abstractmethod, ABC
from typing import Iterable
from random import random, choice
import logging
from math import sqrt, log
from typing import Callable
import numpy as np
import numpy.typing as npt

from rltoolbox.action import Action
from rltoolbox.action_estimate import ActionEstimate


class Policy(ABC):
    def __init__(self, actions: Iterable[Action], initial_expected_reward: float = 0):
        self._actions = tuple(ActionEstimate(a, initial_expected_reward) for a in actions)
        self._steps_made = 0

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def actions(self) -> tuple[ActionEstimate, ...]:
        return self._actions

    @staticmethod
    def get_best_actions(
            action_estimates: tuple[ActionEstimate, ...],
            eval_func: Callable[[ActionEstimate], float] | None = None
        ) -> list[ActionEstimate]:

        a0 = action_estimates[0]
        current_best_reward = a0.expected_reward
        current_best_actions = [a0]

        if eval_func is None:
            eval_func = lambda a_: a_.expected_reward

        for a in action_estimates[1:]:
            v = eval_func(a)
            if v < current_best_reward:
                continue

            if v == current_best_reward:
                current_best_actions.append(a)
            else:
                current_best_reward = v
                current_best_actions = [a]

        return current_best_actions

    @abstractmethod
    def _choose_action(self) -> ActionEstimate:
        pass

    def __call__(self):
        return self._choose_action()

    def make_step(self):
        ae = self._choose_action()
        reward = ae.action.take()
        ae.update(reward)
        self._steps_made += 1
        return reward

    def run(self, n_steps: int) -> npt.NDArray[np.floating]:
        rewards = np.zeros(n_steps)
        for i in range(n_steps):
            rewards[i] = self.make_step()
        return rewards


class GreedyPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _choose_action(self) -> ActionEstimate:
        return choice(self.get_best_actions(self._actions))


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    def _choose_action(self) -> ActionEstimate:
        if random() < self._epsilon:
            self._logger.debug("Choosing action randomly")
            return choice(self._actions)

        self._logger.debug("Choosing from best actions")
        return choice(self.get_best_actions(self._actions))


class UCBPolicy(GreedyPolicy):
    def __init__(self, exploration_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if exploration_rate < 0:
            raise ValueError(f"Exploration rate cannot be negative; got {exploration_rate}")

        if exploration_rate == 0:
            self._logger.warning("With exploration rate = 0, UCB policy falls back to greedy policy")

        self._exploration_rate = exploration_rate

    def get_ucb_value(self, action: ActionEstimate) -> float:
        if not action.times_taken:
            return float('inf')  # maximising action
        return action.expected_reward + self._exploration_rate * sqrt(log(self._steps_made) / action.times_taken)

    def _choose_action(self) -> ActionEstimate:
        return choice(self.get_best_actions(self._actions, eval_func=self.get_ucb_value))
