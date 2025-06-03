from typing import List, Optional
from creature_storage import CreatureStorage
from creature_stats import LAST_DAMAGE_RECEIVED, MASS, MAX_HP, NUM_PRIVATE_FEATURES, NUM_PUBLIC_FEATURES, PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END, PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START, PRIVATE_LINEAR_FEATURES_END, PRIVATE_LINEAR_FEATURES_START, PRIVATE_LOG_FEATURES_END, PRIVATE_LOG_FEATURES_START, PRIVATE_MASS_FRACTION_FEATURES_END, PRIVATE_MASS_FRACTION_FEATURES_START, PRIVATE_MAX_HP_FRACTION_FEATURES_END, PRIVATE_MAX_HP_FRACTION_FEATURES_START, PUBLIC_LOG_FEATURES_END, PUBLIC_LOG_FEATURES_START, PUBLIC_MAX_HP_FRACTION_FEATURES_END, PUBLIC_MAX_HP_FRACTION_FEATURES_START
import numpy as np

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