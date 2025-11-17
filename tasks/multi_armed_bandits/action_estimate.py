from action import Action


class ActionEstimate:
    def __init__(self, action: Action, initial_expected_reward: float = 0.):
        self._action = action
        self._expected_reward = initial_expected_reward
        self._times_taken = 0

    @property
    def action(self) -> Action:
        return self._action

    @property
    def action_name(self) -> str:
        return self._action.name

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
