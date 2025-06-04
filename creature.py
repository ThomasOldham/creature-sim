from typing import List, Optional, Tuple
from creature_storage import CreatureStorage
from creature_stats import BRAIN_MASS, LAST_DAMAGE_RECEIVED, MASS, MAX_HP, MIN_MASS, NUM_PRIVATE_FEATURES, NUM_PUBLIC_FEATURES, PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END, PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START, PRIVATE_LINEAR_FEATURES_END, PRIVATE_LINEAR_FEATURES_START, PRIVATE_LOG_FEATURES_END, PRIVATE_LOG_FEATURES_START, PRIVATE_MASS_FRACTION_FEATURES_END, PRIVATE_MASS_FRACTION_FEATURES_START, PRIVATE_MAX_HP_FRACTION_FEATURES_END, PRIVATE_MAX_HP_FRACTION_FEATURES_START, PUBLIC_LOG_FEATURES_END, PUBLIC_LOG_FEATURES_START, PUBLIC_MAX_HP_FRACTION_FEATURES_END, PUBLIC_MAX_HP_FRACTION_FEATURES_START, UPGRADEABLE_STAT_MASS_CONTRIBUTIONS, UPGRADEABLE_STATS_END, UPGRADEABLE_STATS_START
from network_outputs import ACTION_KINDS_COUNT, PARAMS_MASS_FRACTIONS_END, PARAMS_MASS_FRACTIONS_START, PARAMS_SNAP_DIR_X_END, PARAMS_SNAP_DIR_X_START, PARAMS_SNAP_DIR_Y_END, PARAMS_SNAP_DIR_Y_START, PARAMS_TANH_END, PARAMS_TANH_START
import numpy as np

def recalculate_min_mass(creature_storage: CreatureStorage) -> None:
    stats = creature_storage.stats
    # upgradeable stats contribution
    upgradeable_stats = stats[:, UPGRADEABLE_STATS_START:UPGRADEABLE_STATS_END]
    upgradeable_stats_mass_contributions = upgradeable_stats * UPGRADEABLE_STAT_MASS_CONTRIBUTIONS
    np.sum(upgradeable_stats_mass_contributions, axis=1, out=stats[:, MIN_MASS])
    # brain mass contribution
    np.add(stats[:, MIN_MASS], stats[:, BRAIN_MASS], out=stats[:, MIN_MASS])

_PRIVATE_LINEAR_FEATURE_RANGE_END = PRIVATE_LINEAR_FEATURES_END - PRIVATE_LINEAR_FEATURES_START
_PRIVATE_LOG_FEATURE_RANGE_END = _PRIVATE_LINEAR_FEATURE_RANGE_END + PRIVATE_LOG_FEATURES_END - PRIVATE_LOG_FEATURES_START
_PRIVATE_MASS_FRACTION_FEATURE_RANGE_END = _PRIVATE_LOG_FEATURE_RANGE_END + PRIVATE_MASS_FRACTION_FEATURES_END - PRIVATE_MASS_FRACTION_FEATURES_START
_PRIVATE_MAX_HP_FRACTION_FEATURE_RANGE_END = _PRIVATE_MASS_FRACTION_FEATURE_RANGE_END + PRIVATE_MAX_HP_FRACTION_FEATURES_END - PRIVATE_MAX_HP_FRACTION_FEATURES_START
_PRIVATE_LAST_DAMAGE_FRACTION_FEATURE_RANGE_END = _PRIVATE_MAX_HP_FRACTION_FEATURE_RANGE_END + PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END - PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START

def private_features(creature_storage: CreatureStorage, out: Optional[np.ndarray] = None) -> np.ndarray:
    if out is None:
        out = np.empty((creature_storage.used_row_count(), NUM_PRIVATE_FEATURES), dtype=np.float64)
    linear_features = out[:, :_PRIVATE_LINEAR_FEATURE_RANGE_END]
    linear_features[:] = creature_storage.stats[:, PRIVATE_LINEAR_FEATURES_START:PRIVATE_LINEAR_FEATURES_END]
    log_features = out[:, _PRIVATE_LINEAR_FEATURE_RANGE_END:_PRIVATE_LOG_FEATURE_RANGE_END]
    np.add(creature_storage.stats[:, PRIVATE_LOG_FEATURES_START:PRIVATE_LOG_FEATURES_END], 1.0, out=log_features)
    np.log10(log_features, out=log_features)
    mass_fraction_features = out[:, _PRIVATE_LOG_FEATURE_RANGE_END:_PRIVATE_MASS_FRACTION_FEATURE_RANGE_END]
    np.divide(creature_storage.stats[:, PRIVATE_MASS_FRACTION_FEATURES_START:PRIVATE_MASS_FRACTION_FEATURES_END], creature_storage.stats[:, MASS], out=mass_fraction_features)
    max_hp_fraction_features = out[:, _PRIVATE_MASS_FRACTION_FEATURE_RANGE_END:_PRIVATE_MAX_HP_FRACTION_FEATURE_RANGE_END]
    np.divide(creature_storage.stats[:, PRIVATE_MAX_HP_FRACTION_FEATURES_START:PRIVATE_MAX_HP_FRACTION_FEATURES_END], creature_storage.stats[:, MAX_HP], out=max_hp_fraction_features)
    last_damage_fraction_features = out[:, _PRIVATE_MAX_HP_FRACTION_FEATURE_RANGE_END:_PRIVATE_LAST_DAMAGE_FRACTION_FEATURE_RANGE_END]
    np.divide(creature_storage.stats[:, PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START:PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END], creature_storage.stats[:, LAST_DAMAGE_RECEIVED], out=last_damage_fraction_features)
    return out

_PUBLIC_LOG_FEATURE_RANGE_END = PUBLIC_LOG_FEATURES_END-PUBLIC_LOG_FEATURES_START
_PUBLIC_MAX_HP_FRACTION_FEATURE_RANGE_END = _PUBLIC_LOG_FEATURE_RANGE_END + PUBLIC_MAX_HP_FRACTION_FEATURES_END - PUBLIC_MAX_HP_FRACTION_FEATURES_START

def public_features(creature_storage: CreatureStorage, out: Optional[np.ndarray] = None) -> np.ndarray:
    if out is None:
        out = np.empty((creature_storage.used_row_count(), NUM_PUBLIC_FEATURES), dtype=np.float64)
    log_features = out[:, :_PUBLIC_LOG_FEATURE_RANGE_END]
    np.add(creature_storage.stats[:, PUBLIC_LOG_FEATURES_START:PUBLIC_LOG_FEATURES_END], 1.0, out=log_features)
    np.log10(log_features, out=log_features)
    max_hp_fraction_features = out[:, _PUBLIC_LOG_FEATURE_RANGE_END:_PUBLIC_MAX_HP_FRACTION_FEATURE_RANGE_END]
    np.divide(creature_storage.stats[:, PUBLIC_MAX_HP_FRACTION_FEATURES_START:PUBLIC_MAX_HP_FRACTION_FEATURES_END], creature_storage.stats[:, MAX_HP], out=max_hp_fraction_features)
    return out

DEFAULT_PUBLIC_FEATURES = np.zeros(NUM_PUBLIC_FEATURES, dtype=np.float64)

def transform_features(raw_features_by_vision: List[np.ndarray], creature_storage: CreatureStorage) -> None:
    for (features, features_storage) in zip(raw_features_by_vision, creature_storage.features_storages):
        np.multiply(features, features_storage.feature_coefficients, out=features)
        np.add(features, features_storage.feature_biases, out=features)

def decide_action_kind(network_outputs: np.ndarray) -> np.ndarray:
    normalized_chances = _normalize_action_probabilities(network_outputs[:, :ACTION_KINDS_COUNT])
        
    # Choose actions based on weighted random selection
    return (normalized_chances.cumsum(1) > np.random.rand(normalized_chances.shape[0])[:,None]).argmax(1)

def action_params(network_outputs: np.ndarray, creature_storage: CreatureStorage, out: Optional[np.ndarray] = None) -> np.ndarray:
    if out is None:
        out = np.empty((creature_storage.used_row_count(), network_outputs.PARAMS_COUNT), dtype=np.float64)
    raw_params = network_outputs[:, ACTION_KINDS_COUNT:]

    # Snapped directions
    snapped_xs = out[:, PARAMS_SNAP_DIR_X_START:PARAMS_SNAP_DIR_X_END]
    snapped_ys = out[:, PARAMS_SNAP_DIR_Y_START:PARAMS_SNAP_DIR_Y_END]
    (snapped_xs[:], snapped_ys[:]) = _snap_direction(
        raw_params[:, PARAMS_SNAP_DIR_X_START:PARAMS_SNAP_DIR_X_END],
        raw_params[:, PARAMS_SNAP_DIR_Y_START:PARAMS_SNAP_DIR_Y_END]
    )

    # Tanh
    tanhs = out[:, PARAMS_TANH_START:PARAMS_TANH_END]
    np.tanh(raw_params[:, PARAMS_TANH_START:PARAMS_TANH_END], out=tanhs)

    # Mass fractions
    mass_fractions = out[:, PARAMS_MASS_FRACTIONS_START:PARAMS_MASS_FRACTIONS_END]
    # Apply sigmoid
    np.negative(raw_params[:, PARAMS_MASS_FRACTIONS_START:PARAMS_MASS_FRACTIONS_END], out=mass_fractions)
    np.exp(mass_fractions, out=mass_fractions)
    np.add(mass_fractions, 1.0, out=mass_fractions)
    np.reciprocal(mass_fractions, out=mass_fractions)
    # Multiply by available mass
    np.multiply(mass_fractions, _available_mass(creature_storage), out=mass_fractions)

    # Finally, apply each creature's coefficients
    np.multiply(out, creature_storage.param_coefficients, out=out)
    return out

def _normalize_action_probabilities(action_probabilities: np.ndarray) -> np.ndarray:
    """Normalize action probabilities, handling zero probability cases.
    
    Args:
        action_probabilities: Array of raw action probabilities, one row per creature

    Returns:
        np.ndarray: Normalized probabilities where each row sums to 1
    """
    # Ensure non-negative probabilities
    action_probabilities = np.maximum(0, action_probabilities)
    # Handle zero probability cases
    zero_rows = np.sum(action_probabilities, axis=1) == 0
    action_probabilities[zero_rows] = 1
    # Normalize probabilities
    return action_probabilities / np.sum(action_probabilities, axis=1)[:, np.newaxis]

def _available_mass(creature_storage: CreatureStorage) -> np.ndarray:
    return creature_storage.stats[:, MASS] - creature_storage.stats[:, MIN_MASS]

def _snap_direction(dxs: np.ndarray, dys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Snap directions to cardinal or ordinal directions.
    
    Args:
        dxs: Array of shape (num_creatures, num_directions) containing x-components of directions
        dys: Array of shape (num_creatures, num_directions) containing y-components of directions
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of snapped x and y components, each containing only -1, 0, or 1
    """
    # Get absolute values and determine which component is larger
    abs_dxs = np.abs(dxs)
    abs_dys = np.abs(dys)
    larger = np.maximum(abs_dxs, abs_dys)
    smaller = np.minimum(abs_dxs, abs_dys)
    
    # If smaller component is less than (√2-1) times the larger, snap to cardinal
    cardinal_mask = smaller < 0.414 * larger
    
    # Initialize result arrays
    result_dxs = np.sign(dxs)
    result_dys = np.sign(dys)
    
    # For cardinal directions, zero out the smaller component
    result_dxs[cardinal_mask & (abs_dxs < abs_dys)] = 0
    result_dys[cardinal_mask & (abs_dxs >= abs_dys)] = 0
    
    return result_dxs, result_dys