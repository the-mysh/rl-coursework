import pytest

from cliff_game import CliffGame, Action


@pytest.fixture(scope="function")
def cliff_game():
    yield CliffGame()


@pytest.mark.parametrize(("init_pos", "movement_sequence", "end_pos"), (
    ((2, 3), [Action.UP], (1, 3)),
    ((2, 3), [Action.DOWN], (3, 3)),
    ((2, 3), [Action.RIGHT], (2, 4)),
    ((2, 3), [Action.LEFT], (2, 2)),
    ((1, 0), [Action.UP, Action.RIGHT, Action.RIGHT], (0, 2)),
    ((3, 8), [Action.LEFT, Action.RIGHT, Action.LEFT, Action.UP, Action.LEFT], (2, 6)),
    ((0, 0), [Action.UP], (0, 0)),
    ((2, 10), [Action.RIGHT, Action.RIGHT, Action.DOWN, Action.RIGHT, Action.RIGHT], (3, 11)),
    ((2, 10), [Action.RIGHT, Action.RIGHT, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.LEFT], (3, 10)),
    ((3, 2), [Action.DOWN, Action.DOWN, Action.LEFT, Action.DOWN, Action.LEFT, Action.LEFT, Action.LEFT], (3, 0))
))
def test_movement(cliff_game, init_pos, movement_sequence, end_pos):
    cliff_game._agent_pos = init_pos
    for move in movement_sequence:
        cliff_game.move(move)
    assert cliff_game._agent_pos == end_pos
