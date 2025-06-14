from typing import Tuple
import numpy as np
from board import Board
from creature_storage import CreatureStorage
from execution_timer import timer_decorator
from action_kind import (
    ACTION_ATTACK, ACTION_EAT, ACTION_HEAL, ACTION_UPGRADE, ACTION_MOVE, ACTION_REPRODUCE
)
from network_outputs import (
    PARAM_REPRODUCE_DX, PARAM_REPRODUCE_DY, PARAM_ATTACK_DX, PARAM_ATTACK_DY,
    PARAM_MOVE_DX, PARAM_MOVE_DY, PARAM_OFFSPRING_MASS_FRACTION, PARAM_HEAL_MASS_FRACTION,
    PARAMS_UPGRADE_MASS_FRACTIONS_START, PARAMS_UPGRADE_MASS_FRACTIONS_END
)
from creature_stats import EAT_RATE, MOVE_RATE, MASS, SUB_X, SUB_Y, UPGRADEABLE_STATS_START, UPGRADEABLE_STATS_END, UPGRADE_COSTS, HEAL_POWER, DAMAGE, ATTACK_POWER, LAST_DAMAGE_RECEIVED, LAST_DAMAGE_DX_SUM, LAST_DAMAGE_DY_SUM
from genome import Genome, indel_suppression_tax_curve, INTERVAL_SUPPRESSION_TAX_FACTORS, POINT_MUTATION_SUPPRESSION_TAX_FACTOR

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
    # Get attacker positions and calculate target positions
    attacker_positions = creature_storage.grid_position[mask]
    attack_dx = params[mask, PARAM_ATTACK_DX]
    attack_dy = params[mask, PARAM_ATTACK_DY]
    target_positions = attacker_positions + np.column_stack((attack_dx, attack_dy))
    
    # Set border to -1 to treat edge cells as unoccupied
    for border in board.border_pseudocreatures:
        border.fill(-1)
    
    # Get target creatures (if any), including border
    target_indices = board.creatures_with_border[target_positions[:, 1] + 1, target_positions[:, 0] + 1]
    
    # Skip attacks where target is unoccupied (-1) or is self (dx and dy both 0)
    valid_attack_mask = (target_indices >= 0) & ~((attack_dx == 0) & (attack_dy == 0))
    
    # Get attack powers for valid attacks
    attack_powers = creature_storage.stats[mask, ATTACK_POWER][valid_attack_mask]
    
    # Apply damage to valid targets
    valid_targets = target_indices[valid_attack_mask]
    creature_storage.stats[valid_targets, DAMAGE] += attack_powers
    creature_storage.stats[valid_targets, LAST_DAMAGE_RECEIVED] += attack_powers
    
    # Update damage direction sums for valid targets
    valid_attack_dx = attack_dx[valid_attack_mask]
    valid_attack_dy = attack_dy[valid_attack_mask]
    creature_storage.stats[valid_targets, LAST_DAMAGE_DX_SUM] -= valid_attack_dx
    creature_storage.stats[valid_targets, LAST_DAMAGE_DY_SUM] -= valid_attack_dy
    
    # Calculate success (1.0 for valid attacks, 0.0 for skipped)
    success = np.zeros_like(mask, dtype=np.float64)
    success[mask] = valid_attack_mask.astype(np.float64)
    
    # Set result values
    out[mask, RESULT_SUCCESS] = success[mask]
    out[mask, RESULT_COST] = 0.0  # No cost for attacking
    out[mask, RESULT_KIND] = ACTION_ATTACK
    out[mask, RESULT_DIR_X] = attack_dx
    out[mask, RESULT_DIR_Y] = attack_dy
    
    board.check_and_apply_killings()
    
    return out

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
    mass_fractions = params[mask, PARAMS_UPGRADE_MASS_FRACTIONS_START:PARAMS_UPGRADE_MASS_FRACTIONS_END]
    upgrade_amounts = mass_fractions / UPGRADE_COSTS
    creature_storage.stats[mask, UPGRADEABLE_STATS_START:UPGRADEABLE_STATS_END] += upgrade_amounts
    
    # Calculate success (1.0 if any upgrade amount was positive, 0.0 if all were 0)
    success = np.any(upgrade_amounts > 0, axis=1).astype(np.float64)
    
    # Set result values
    out[mask, RESULT_SUCCESS] = success
    out[mask, RESULT_COST] = np.sum(mass_fractions, axis=1)  # Total mass spent
    out[mask, RESULT_KIND] = ACTION_UPGRADE
    out[mask, RESULT_DIR_X] = 0.0  # No direction for upgrading
    out[mask, RESULT_DIR_Y] = 0.0  # No direction for upgrading
    
    return out

MAX_MOVE_COST_PER_DIMENSION = 0.1
MAX_SUB_POSITION = np.float64(1.0 - (10.0 ** -12))

@timer_decorator('action._handle_grid_movement')
def _handle_grid_movement(mask: np.ndarray, creature_storage: CreatureStorage, board: Board) -> np.ndarray:
    """Handle grid movement for creatures whose sub-positions are out of bounds.
    
    Args:
        mask: Mask of creatures attempting grid movement
        creature_storage: Storage containing creature data
        board: The game board
        
    Returns:
        Mask indicating which creatures actually moved grid positions
    """
    # Get current positions and sub-positions
    current_positions = creature_storage.grid_position[mask]
    sub_positions = creature_storage.stats[mask][:, [SUB_X, SUB_Y]]
    
    # Calculate desired new grid positions, clamping to board bounds
    desired_positions = current_positions + np.floor(sub_positions).astype(np.int64)
    desired_positions[:, 0] = np.clip(desired_positions[:, 0], 0, board.width - 1)
    desired_positions[:, 1] = np.clip(desired_positions[:, 1], 0, board.height - 1)
    
    # Get target cells (if any)
    target_indices = board.creatures[desired_positions[:, 1], desired_positions[:, 0]]
    
    # Skip moves where target is occupied
    valid_move_mask = target_indices < 0
    
    # For each target position, find the highest mass creature targeting it
    valid_targets = desired_positions[valid_move_mask]
    valid_masses = creature_storage.stats[mask, MASS][valid_move_mask]
    
    # Select highest mass mover for each target
    final_mover_indices, final_targets = _select_actor_per_target(valid_targets, valid_masses)
    
    # Get the original indices of the movers
    original_indices = np.where(mask)[0][valid_move_mask][final_mover_indices]
    
    # Calculate grid deltas for successful moves
    grid_deltas = final_targets - creature_storage.grid_position[original_indices]
    
    # Update grid positions for successful moves
    creature_storage.grid_position[original_indices] = final_targets
    board.creatures[final_targets[:, 1], final_targets[:, 0]] = original_indices
    board.creatures[current_positions[valid_move_mask][final_mover_indices, 1],
                   current_positions[valid_move_mask][final_mover_indices, 0]] = -1
    
    # Adjust sub-positions for creatures that moved grid positions
    creature_storage.stats[original_indices, SUB_X] -= grid_deltas[:, 0]
    creature_storage.stats[original_indices, SUB_Y] -= grid_deltas[:, 1]
    
    # Create mask of which creatures actually moved grid positions
    grid_moved_mask = np.zeros_like(mask, dtype=bool)
    grid_moved_mask[original_indices] = True
    
    return grid_moved_mask

@timer_decorator('action.move_action')
def move_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                board: Board, out: np.ndarray) -> np.ndarray:
    # Get base movement directions and calculate intended movements
    base_dx = params[mask, PARAM_MOVE_DX]
    base_dy = params[mask, PARAM_MOVE_DY]
    move_rates = creature_storage.stats[mask, MOVE_RATE]
    intended_dx = base_dx * move_rates
    intended_dy = base_dy * move_rates
    
    # Store initial positions for success calculation
    initial_grid_positions = creature_storage.grid_position[mask].copy()
    initial_sub_positions = creature_storage.stats[mask][:, [SUB_X, SUB_Y]].copy()
    
    # Update sub-positions
    creature_storage.stats[mask, SUB_X] += intended_dx
    creature_storage.stats[mask, SUB_Y] += intended_dy
    
    # Find creatures trying to move grid positions
    grid_move_mask = (creature_storage.stats[mask, SUB_X] >= 1.0) | \
                    (creature_storage.stats[mask, SUB_Y] >= 1.0) | \
                    (creature_storage.stats[mask, SUB_X] < 0.0) | \
                    (creature_storage.stats[mask, SUB_Y] < 0.0)
    
    # Handle grid movement
    grid_moved_mask = _handle_grid_movement(grid_move_mask, creature_storage, board)
    
    # Clamp sub-positions
    creature_storage.stats[mask, SUB_X] = np.clip(
        creature_storage.stats[mask, SUB_X], 0.0, MAX_SUB_POSITION)
    creature_storage.stats[mask, SUB_Y] = np.clip(
        creature_storage.stats[mask, SUB_Y], 0.0, MAX_SUB_POSITION)
    
    # Calculate actual movement
    final_grid_positions = creature_storage.grid_position[mask]
    final_sub_positions = creature_storage.stats[mask][:, [SUB_X, SUB_Y]]
    actual_dx = (final_grid_positions[:, 0] + final_sub_positions[:, 0]) - \
                (initial_grid_positions[:, 0] + initial_sub_positions[:, 0])
    actual_dy = (final_grid_positions[:, 1] + final_sub_positions[:, 1]) - \
                (initial_grid_positions[:, 1] + initial_sub_positions[:, 1])
    
    # Calculate success
    actual_distance = np.abs(actual_dx) + np.abs(actual_dy)
    intended_distance = np.abs(intended_dx) + np.abs(intended_dy)
    intended_distance[intended_distance == 0.0] = 1.0  # Avoid division by zero
    success = actual_distance / intended_distance
    
    # Calculate cost
    cost = MAX_MOVE_COST_PER_DIMENSION * creature_storage.stats[mask, MASS] * \
           (np.abs(base_dx) + np.abs(base_dy))
    
    # Set result values
    out[mask, RESULT_SUCCESS] = success
    out[mask, RESULT_COST] = cost
    out[mask, RESULT_KIND] = ACTION_MOVE
    out[mask, RESULT_DIR_X] = base_dx
    out[mask, RESULT_DIR_Y] = base_dy
    
    return out

@timer_decorator('action._fidelity_tax')
def _fidelity_tax(genome: Genome) -> float:
    """Calculate the fidelity tax for a genome based on its mutation control parameters."""
    point_tax = POINT_MUTATION_SUPPRESSION_TAX_FACTOR * genome.mutation_control.average_mutsup
    interval_taxes = indel_suppression_tax_curve(genome.mutation_control.interval_mutation_rates) * INTERVAL_SUPPRESSION_TAX_FACTORS
    return genome.size() * (point_tax + np.sum(interval_taxes))

@timer_decorator('action._select_actor_per_target')
def _select_actor_per_target(target_positions: np.ndarray, priorities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Select the highest priority actor for each target position.
    
    Args:
        target_positions: Array of target positions, shape (n, 2)
        priorities: Array of priorities for each actor, shape (n,)
        
    Returns:
        Tuple of (selected_actor_indices, selected_targets) where:
        - selected_actor_indices are the indices of the actors that won their targets
        - selected_targets are the corresponding target positions
    """
    # If no targets, return early with correct shapes
    if len(target_positions) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    
    # Create a dictionary to track highest priority for each target
    target_to_priority = {}
    target_to_actor = {}
    for i, (target, priority) in enumerate(zip(target_positions, priorities)):
        target_key = tuple(target)
        if target_key not in target_to_priority or priority > target_to_priority[target_key]:
            target_to_priority[target_key] = priority
            target_to_actor[target_key] = i
    
    # Convert results to arrays
    selected_targets = np.array(list(target_to_actor.keys()), dtype=np.int64)
    selected_actors = np.array(list(target_to_actor.values()), dtype=np.int64)
    
    return selected_actors, selected_targets

@timer_decorator('action.reproduce_action')
def reproduce_action(mask: np.ndarray, params: np.ndarray, creature_storage: CreatureStorage,
                     board: Board, out: np.ndarray) -> np.ndarray:
    # Get reproducer positions and calculate target positions
    reproducer_positions = creature_storage.grid_position[mask]
    reproduce_dx = params[mask, PARAM_REPRODUCE_DX]
    reproduce_dy = params[mask, PARAM_REPRODUCE_DY]
    target_positions = reproducer_positions + np.column_stack((reproduce_dx, reproduce_dy))
    
    # Set border to non-negative to treat edge cells as occupied
    for border in board.border_pseudocreatures:
        border.fill(12345)
    
    # Get target cells (if any), including border
    target_cells = board.creatures_with_border[target_positions[:, 1] + 1, target_positions[:, 0] + 1]
    
    # Skip reproductions where target is occupied (including edge cells)
    valid_reproduce_mask = target_cells < 0
    
    # Get mass spent for each reproduction
    mass_spent = params[mask, PARAM_OFFSPRING_MASS_FRACTION]
    
    # For each target position, find the highest mass spent reproduction targeting it
    valid_targets = target_positions[valid_reproduce_mask]
    valid_masses = mass_spent[valid_reproduce_mask]
    
    # Select highest mass spent reproduction for each target
    final_reproducer_indices, final_targets = _select_actor_per_target(valid_targets, valid_masses)
    
    # Get the original indices of the reproducers
    original_indices = np.where(mask)[0][valid_reproduce_mask][final_reproducer_indices]
    
    # Get the masses for the selected reproductions
    final_masses = valid_masses[final_reproducer_indices]
    
    # Perform the reproductions
    for i, (target, mass, reproducer_idx) in enumerate(zip(final_targets, final_masses, original_indices)):
        # Create and mutate offspring genome
        parent_genome = creature_storage.genome[reproducer_idx]
        offspring_genome = parent_genome.deep_copy()
        offspring_genome.mutate()
        
        # Add the offspring creature
        offspring_idx = board.add_creature(offspring_genome)
        reproducer_debug_info = creature_storage.debug_info[reproducer_idx]
        creature_storage.debug_info[offspring_idx].ancestors = reproducer_debug_info.ancestors + [reproducer_debug_info.id]
        
        # Set offspring position
        creature_storage.grid_position[offspring_idx] = target
        board.creatures[target[1], target[0]] = offspring_idx
        
        # Calculate and apply fidelity tax
        fidelity_tax = _fidelity_tax(parent_genome)
        creature_storage.stats[offspring_idx, MASS] = mass - fidelity_tax
    
    # Calculate success (1.0 for successful reproductions, 0.0 for skipped)
    success = np.zeros_like(mask, dtype=np.float64)
    success[mask] = valid_reproduce_mask.astype(np.float64)
    
    # Set result values
    out[mask, RESULT_SUCCESS] = success[mask]
    out[mask, RESULT_COST] = mass_spent
    out[mask, RESULT_KIND] = ACTION_REPRODUCE
    out[mask, RESULT_DIR_X] = reproduce_dx
    out[mask, RESULT_DIR_Y] = reproduce_dy
    
    return out

ACTION_FUNCTIONS = [
    attack_action,
    eat_action,
    heal_action,
    upgrade_action,
    move_action,
    reproduce_action,
]