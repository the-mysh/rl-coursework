from random import gauss
import logging
import numpy as np
import numpy.typing as npt

from rltoolbox.action import Action
from rltoolbox.policy import GreedyPolicy, EpsilonGreedyPolicy, UCBPolicy, Policy


logger = logging.getLogger(__name__)


def define_actions(n: int) -> list[Action]:
    return [Action(str(i), gauss()) for i in range(n)]


def define_policies(actions: list[Action]) -> dict[str, Policy]:
    policies = {
        "greedy r0=0": GreedyPolicy(actions),
        "greedy r0=5": GreedyPolicy(actions, initial_expected_reward=5),
        "epsilon-greedy e=0.1": EpsilonGreedyPolicy(epsilon=0.1, actions=actions),
        "epsilon-greedy e=0.01": EpsilonGreedyPolicy(epsilon=0.01, actions=actions),
        "UCB": UCBPolicy(exploration_rate=0.1, actions=actions),
    }

    return policies


def run_policies(policies: dict[str, Policy], n_steps: int) -> dict[str, npt.NDArray[np.floating]]:
    results = {}
    for policy_name, policy in policies.items():
        logger.debug(f"Running policy {policy_name} for {n_steps} steps")
        results[policy_name] = policy.run(n_steps)

    return results


def main():
    actions = define_actions(10)
    policies = define_policies(actions)


    t = max(len(p) for p in policies)
    t = (t // 4 + 1) * 4

    for i in range(10):
        print(f"Iteration {i}:")
        for policy_name, policy in policies.items():
            print(f"\t{policy_name}:{(t-len(policy_name))*' '}action {policy().action_name}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
