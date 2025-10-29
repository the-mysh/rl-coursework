from typing import Iterable


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


class Policy:
    def __init__(self, action_names: Iterable[str], initial_expected_reward: float = 0):
        self._actions = tuple(Action(name, initial_expected_reward) for name in action_names)

    @property
    def actions(self) -> tuple[Action, ...]:
        return self._actions


if __name__ == '__main__':
    p = Policy([str(i) for i in range(5)])
    for a in p.actions:
        print(a)

