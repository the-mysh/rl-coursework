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
    ((0, 8), [Action.LEFT, Action.RIGHT, Action.LEFT, Action.UP, Action.LEFT], (0, 6)),
    ((0, 0), [Action.UP], (0, 0)),
    ((2, 10), [Action.RIGHT, Action.RIGHT, Action.UP, Action.RIGHT, Action.RIGHT], (1, 11)),
    ((2, 0), [Action.LEFT, Action.LEFT, Action.DOWN, Action.LEFT, Action.DOWN, Action.DOWN], (3, 0))
))
def test_movement(cliff_game, init_pos, movement_sequence, end_pos):
    cliff_game._agent_pos = init_pos
    for move in movement_sequence:
        cliff_game.move(move)
    assert cliff_game._agent_pos == end_pos


@pytest.mark.parametrize(("init_pos", "action", "reward", "game_over"), (
    ((0, 0), Action.UP, -1, False),
    ((1, 0), Action.UP, -1, False),
    ((2, 5), Action.RIGHT, -1, False),
    ((2, 5), Action.LEFT, -1, False),
    ((2, 5), Action.DOWN, -100, True),
    ((2, 0), Action.DOWN, -1, False),
    ((3, 0), Action.RIGHT, -100, True),
    ((3, 0), Action.LEFT, -1, False),
    ((2, 11), Action.DOWN, -1, True),
))
def test_reward_and_game_over(cliff_game, init_pos, action, reward, game_over):
    cliff_game._agent_pos = init_pos
    r, over = cliff_game.move(action)
    assert r == reward
    assert over is game_over


@pytest.mark.parametrize("pos", ((3, 2), (3, 10), (3, 11)))
def test_game_over_error_and_reset(cliff_game, pos):
    cliff_game._agent_pos = pos

    with pytest.raises(RuntimeError, match="Game is over, reset"):
        cliff_game.move(Action.RIGHT)

    cliff_game.reset()
    cliff_game.move(Action.RIGHT)  # no error
