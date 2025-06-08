from typing import Optional
import numpy as np
from board import Board
from creature_storage import CreatureStorage
from execution_timer import timer_decorator
from network_outputs import ACTION_ATTACK, ACTION_EAT, ACTION_HEAL, ACTION_UPGRADE, ACTION_MOVE, ACTION_REPRODUCE
from creature_stats import EAT_RATE, MASS

RESULT_SUCCESS = 0
RESULT_COST = 1
RESULT_KIND = 2
RESULT_DIR_X = 3
RESULT_DIR_Y = 4
RESULT_SIZE = 5

# TODO
@timer_decorator('action.attack_action')
def attack_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                   board: Board, out: np.ndarray) -> np.ndarray:
    return np.zeros((creature_storage.used_row_count(), RESULT_SIZE), dtype=np.float64)

# TODO
@timer_decorator('action.eat_action')
def eat_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
               board: Board, out: np.ndarray) -> np.ndarray:
    positions = creature_storage.grid_position[mask]
    eat_rates = creature_storage.stats[mask, EAT_RATE]
    food_amounts = board.food[positions[:, 1], positions[:, 0]]
    amounts_eaten = np.minimum(eat_rates, food_amounts)
    
    # Subtract eaten food from board and add eaten food to creature masses
    board.food[positions[:, 1], positions[:, 0]] -= amounts_eaten
    creature_storage.stats[mask, MASS] += amounts_eaten
    
    out[mask, RESULT_SUCCESS] = amounts_eaten / eat_rates  # Success is fraction of eat_rate achieved
    out[mask, RESULT_COST] = 0.0  # No cost for eating
    out[mask, RESULT_KIND] = ACTION_EAT
    out[mask, RESULT_DIR_X] = 0.0  # No direction for eating
    out[mask, RESULT_DIR_Y] = 0.0  # No direction for eating
    
    return out

# TODO
@timer_decorator('action.heal_action')
def heal_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: np.ndarray) -> np.ndarray:
    return np.zeros((creature_storage.used_row_count(), RESULT_SIZE), dtype=np.float64)

# TODO
@timer_decorator('action.upgrade_action')
def upgrade_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                   board: Board, out: np.ndarray) -> np.ndarray:
    return np.zeros((creature_storage.used_row_count(), RESULT_SIZE), dtype=np.float64)

# TODO
@timer_decorator('action.move_action')
def move_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: np.ndarray) -> np.ndarray:
    return np.zeros((creature_storage.used_row_count(), RESULT_SIZE), dtype=np.float64)

# TODO
@timer_decorator('action.reproduce_action')
def reproduce_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                     board: Board, out: np.ndarray) -> np.ndarray:
    return np.zeros((creature_storage.used_row_count(), RESULT_SIZE), dtype=np.float64)

ACTION_FUNCTIONS = [
    attack_action,
    eat_action,
    heal_action,
    upgrade_action,
    move_action,
    reproduce_action,
]