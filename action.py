from typing import Optional
import numpy as np

from board import Board
from creature_storage import CreatureStorage

RESULT_SUCCESS = 0
RESULT_COST = 1
RESULT_KIND = 2
RESULT_DIR_X = 3
RESULT_DIR_Y = 4
RESULT_SIZE = 5

def attack_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                   board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

def eat_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
               board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

def heal_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

def upgrade_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                   board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

def move_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: Optional[np.ndarray] = None) -> np.ndarray:
    pass # TODO

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