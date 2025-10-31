from utils import GamblerPolicy


def test_initial_ptpm():
    gp = GamblerPolicy(goal=5, success_probability=0.2)
    ptpm = gp.define_ptpm()

    assert ptpm.shape == (4, 6)
    assert (r.sum() == 1 for r in ptpm)

    assert ptpm[0, 0] == 0.8
    assert ptpm[0, 2] == 0.2

    assert ptpm[1, 1] == 0.8
    assert ptpm[1, 3] == 0.2

    assert ptpm[3, 3] == 0.8
    assert ptpm[3, 5] == 0.2
