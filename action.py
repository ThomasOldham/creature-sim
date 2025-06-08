from typing import Optional
import numpy as np
from board import Board
from creature_storage import CreatureStorage
from execution_timer import timer_decorator
from network_outputs import (
    ACTION_ATTACK, ACTION_EAT, ACTION_HEAL, ACTION_UPGRADE, ACTION_MOVE, ACTION_REPRODUCE,
    PARAM_REPRODUCE_DX, PARAM_REPRODUCE_DY, PARAM_ATTACK_DX, PARAM_ATTACK_DY,
    PARAM_MOVE_DX, PARAM_MOVE_DY, PARAM_OFFSPRING_MASS_FRACTION, PARAM_HEAL_MASS_FRACTION,
    PARAMS_UPGRADE_MASS_FRACTIONS_START, PARAMS_UPGRADE_MASS_FRACTIONS_END
)
from creature_stats import EAT_RATE, MASS, NUM_UPGRADEABLE_STATS, HEAL_POWER, DAMAGE

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
    # Get mass spent and heal power to calculate healing amount
    heal_powers = creature_storage.stats[mask, HEAL_POWER]
    mass_spent = params[mask, PARAM_HEAL_MASS_FRACTION]
    healing_amounts = np.power(mass_spent, 0.9) * heal_powers
    
    # Calculate actual damage reduction (can't reduce below 0)
    current_damage = creature_storage.stats[mask, DAMAGE]
    damage_reduction = np.minimum(current_damage, healing_amounts)
    
    # Apply healing
    creature_storage.stats[mask, DAMAGE] -= damage_reduction
    creature_storage.stats[mask, MASS] -= mass_spent
    
    # Calculate success
    # If no healing occurred, success is 0
    # Otherwise, success is ratio of potential healing to actual healing
    damage_reduction[damage_reduction <= 0.0] = 1.0
    success = healing_amounts / damage_reduction
    
    # Set result values
    out[mask, RESULT_SUCCESS] = success
    out[mask, RESULT_COST] = mass_spent
    out[mask, RESULT_KIND] = ACTION_HEAL
    out[mask, RESULT_DIR_X] = 0.0  # No direction for healing
    out[mask, RESULT_DIR_Y] = 0.0  # No direction for healing
    
    return out

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