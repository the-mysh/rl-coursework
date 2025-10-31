import numpy as np
import numpy.testing as np_test

from utils import GamblerProblemModel


def test_initial_ptpm():
    gp = GamblerProblemModel(goal=5, success_probability=0.2)
    ptpm = gp.define_ptpm(gp.propose_policy_estimate())

    assert ptpm.shape == (6, 6)
    assert (r.sum() == 1 for r in ptpm[1:-1])
    assert ptpm[0].sum() == 0
    assert ptpm[-1].sum() == 0

    assert ptpm[1, 0] == 0.8
    assert ptpm[1, 2] == 0.2

    assert ptpm[2, 1] == 0.8
    assert ptpm[2, 3] == 0.2

    assert ptpm[3, 2] == 0.8
    assert ptpm[3, 4] == 0.2


def test_immediate_rewards():
    gp = GamblerProblemModel(goal=8, success_probability=0.2)
    np_test.assert_array_equal(gp.immediate_rewards, np.array(8 * [0] + [1]).astype(float))
