from random import gauss


class Action:
    def __init__(self, name: str, average_reward: float):
        self._name = name
        self._average_reward = average_reward

    @property
    def name(self) -> str:
        return self._name

    def take(self):
        reward = gauss(self._average_reward)
        return reward
