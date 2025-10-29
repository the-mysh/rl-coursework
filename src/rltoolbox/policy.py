from abc import abstractmethod, ABC
from typing import Iterable
from random import random, choice
import logging
from math import sqrt, log
from typing import Callable


logger = logging.getLogger(__name__)


class Action:
    def __init__(self, name: str, initial_expected_reward: float = 0.):
        self._name = name
        self._expected_reward = initial_expected_reward
        self._times_taken = 0

    def __str__(self) -> str:
        return f"Action '{self._name}', taken {self._times_taken} times, expected reward: {self._expected_reward}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def expected_reward(self) -> float:
        return self._expected_reward

    @property
    def times_taken(self) -> int:
        return self._times_taken

    @property
    def step_size(self) -> float:
        if not self._times_taken:
            raise RuntimeError("Step size not defined before action is taken at least once")
        return 1 / self._times_taken

    def update(self, received_reward: float):
        self._times_taken += 1
        self._expected_reward = self._expected_reward + self.step_size * (received_reward - self._expected_reward)


class Policy(ABC):
    def __init__(self, action_names: Iterable[str], initial_expected_reward: float = 0):
        self._actions = tuple(Action(name, initial_expected_reward) for name in action_names)
        self._steps_made = 0

    @property
    def actions(self) -> tuple[Action, ...]:
        return self._actions

    @staticmethod
    def get_best_actions(actions: tuple[Action, ...], eval_func: Callable[[Action], float] | None = None
                         ) -> list[Action]:
        a0 = actions[0]
        current_best_reward = a0.expected_reward
        current_best_actions = [a0]

        if eval_func is None:
            eval_func = lambda a_: a_.expected_reward

        for a in actions[1:]:
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
    def _choose_action(self) -> Action:
        pass

    def __call__(self):
        return self._choose_action()


class GreedyPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _choose_action(self) -> Action:
        return choice(self.get_best_actions(self._actions))


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _choose_action(self) -> Action:
        if random() < self._epsilon:
            self._logger.debug("Choosing action randomly")
            return choice(self._actions)

        self._logger.debug("Choosing from best actions")
        return choice(self.get_best_actions(self._actions))


class UCBPolicy(GreedyPolicy):
    def __init__(self, exploitation_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exploitation_rate = exploitation_rate

    def get_ucb_value(self, action: Action) -> float:
        if not action.times_taken:
            return action.expected_reward
        return action.expected_reward + self._exploitation_rate * sqrt(log(self._steps_made) / action.times_taken)

    def _choose_action(self) -> Action:
        return choice(self.get_best_actions(self._actions, eval_func=self.get_ucb_value))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    ans = [str(i) for i in range(5)]
    gp0 = GreedyPolicy(ans)
    gp5 = GreedyPolicy(ans, initial_expected_reward=5)
    egp = EpsilonGreedyPolicy(epsilon=0.1, action_names=ans)
    ucb = UCBPolicy(exploitation_rate=0.1, action_names=ans)

    for i in range(10):
        print(f"Iteration {i}:\n\tGP0: {gp0().name};\n\tGP5: {gp5().name};\n\tEGP: {egp().name}\n\tUCB: {ucb().name}")

