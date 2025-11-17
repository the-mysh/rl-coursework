import numpy as np
from enum import Enum, auto


class SquareType(Enum):
    START = auto()
    GOAL = auto()
    CLIFF = auto()
    THROUGH = auto()


class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class CliffGame:
    def __init__(self):
        scene_shape = (4, 12)
        start_pos = (-1, 0)
        goal_pos = (-1, -1)
        self._scene = self.define_scene(*scene_shape, start_pos=start_pos, goal_pos=goal_pos)

        self._start_pos = self.translate_pos(start_pos)
        self._goal_pos = self.translate_pos(goal_pos)

        self.square_rep = {
            SquareType.CLIFF: 'C',
            SquareType.START: 'S',
            SquareType.GOAL: 'G',
            SquareType.THROUGH: ' '
        }

        self.movement = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }

        self._agent_pos = self.start_pos[:]

    @property
    def scene(self):
        return self._scene

    @property
    def start_pos(self):
        return self._start_pos

    @property
    def goal_pos(self):
        return self._goal_pos

    @property
    def agent_pos(self):
        return self._agent_pos

    @property
    def agent_square(self):
        return self._scene[*self.agent_pos]

    def reset(self):
        self._agent_pos = self.start_pos[:]

    def move(self, action: Action) -> tuple[int, bool]:
        if not isinstance(action, Action):
            raise TypeError(f"Expected an Action enum; got {type(action)}: {action}")

        if self.agent_square is SquareType.CLIFF or self.agent_square is SquareType.GOAL:
            raise RuntimeError("Game is over, reset to play again")

        agent_x, agent_y = self._agent_pos
        move_x, move_y = self.movement[action]

        agent_x = agent_x + move_x
        agent_y = agent_y + move_y

        self._agent_pos = self.bind_pos((agent_x, agent_y))

        a = self.agent_square
        if a is SquareType.CLIFF:
            return -100, True
        return -1, (a is SquareType.GOAL)

    def bind_pos(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        sx, sy = self._scene.shape
        x = min(max(x, 0), sx - 1)
        y = min(max(y, 0), sy - 1)
        return x, y

    def translate_pos(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        sx, sy = self._scene.shape
        if x < 0:
            x = sx + x
        if y < 0:
            y = sy + y
        return x, y

    @staticmethod
    def define_scene(depth: int, width: int, start_pos: tuple[int, int], goal_pos: tuple[int, int]):
        scene = np.full((depth, width), fill_value=SquareType.THROUGH)
        scene[-1, :] = SquareType.CLIFF
        scene[*start_pos] = SquareType.START
        scene[*goal_pos] = SquareType.GOAL

        return scene

    def __str__(self):

        row_reps = []
        separator_row = "\n" + "|-----" * self._scene.shape[1] + "|\n"

        for ri, row in enumerate(self._scene):
            cell_reps = []
            for ci, cell in enumerate(row):
                v = self.square_rep[cell]
                if self._agent_pos == (ri, ci):
                    v += "*"
                v = "  " + v + (3-len(v)) * " "
                cell_reps.append(v)
            row_reps.append('|' +  '|'.join(cell_reps) + '|')

        s = separator_row + separator_row.join(row_reps) + separator_row
        return s
