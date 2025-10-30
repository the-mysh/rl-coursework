from random import gauss


class Action:
    def __init__(self, name: str, average_reward: float):
        self._name = name
        self._average_reward = average_reward

    @property
    def name(self) -> str:
        return self._name

    @property
    def average_reward(self) -> float:
        return self._average_reward

    def take(self):
        reward = gauss(self._average_reward)
        return reward
