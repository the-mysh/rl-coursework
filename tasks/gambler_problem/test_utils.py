from utils import GamblerPolicy


def test_initial_ptpm():
    gp = GamblerPolicy(goal=5, success_probability=0.2)
    ptpm = gp.define_ptpm()

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
