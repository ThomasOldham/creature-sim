from typing import Optional
import numpy as np

from board import Board
from creature_storage import CreatureStorage
from execution_timer import timer_decorator

RESULT_SUCCESS = 0
RESULT_COST = 1
RESULT_KIND = 2
RESULT_DIR_X = 3
RESULT_DIR_Y = 4
RESULT_SIZE = 5

@timer_decorator('action.attack_action')
def attack_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                   board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

@timer_decorator('action.eat_action')
def eat_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
               board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

@timer_decorator('action.heal_action')
def heal_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

@timer_decorator('action.upgrade_action')
def upgrade_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                   board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

@timer_decorator('action.move_action')
def move_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

@timer_decorator('action.reproduce_action')
def reproduce_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                     board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

ACTION_FUNCTIONS = [
    attack_action,
    eat_action,
    heal_action,
    upgrade_action,
    move_action,
    reproduce_action,
]