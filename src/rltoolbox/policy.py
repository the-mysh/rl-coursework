from abc import abstractmethod, ABC
from typing import Iterable
from random import choice


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

    @property
    def actions(self) -> tuple[Action, ...]:
        return self._actions

    def get_best_actions(self):
        a0 = self._actions[0]
        current_best_reward = a0.expected_reward
        current_best_actions = [a0]

        for a in self._actions[1:]:
            if (r := a.expected_reward) < current_best_reward:
                continue

            if r == current_best_reward:
                current_best_actions.append(a)
            else:
                current_best_reward = r
                current_best_actions = [a]

        return current_best_actions

    @abstractmethod
    def _choose_action(self, best_actions: Iterable[Action]) -> Action:
        pass

    def __call__(self):
        best_actions = self.get_best_actions()
        return self._choose_action(best_actions)


class GreedyPolicy(Policy):
    def __init__(self, action_names: Iterable[str]):
        super().__init__(action_names)

    def _choose_action(self, best_actions: list[Action]) -> Action:
        return choice(best_actions)


if __name__ == '__main__':
    p = GreedyPolicy([str(i) for i in range(5)])

    for i in range(10):
        print(f"Iteration {i} - choosing {p()}")

