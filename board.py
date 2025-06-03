from typing import Optional
from creature_storage import CreatureStorage
from genome import Genome
import cell_stats
import creature
import creature_stats
import numpy as np

class Board:
    def __init__(self, width: int, height: int):
        self.width = width # set at construction and fixed for the lifetime of the board
        self.height = height # set at construction and fixed for the lifetime of the board
        self.food = np.zeros((width, height), dtype=np.float64) # food amounts on the board (should never be negative)
        self.creatures_with_border = np.full((width + 2, height + 2), -1.0, dtype=np.int64) # view of creatures with a border manipulable through border_pseudocreatures
        self.creatures = self.creatures_with_border[1:-1, 1:-1] # indices of creatures on the board
        self.border_pseudocreatures = [ # mutable interface for arbitrarily setting border conditions to help handle edge cases
            self.creatures_with_border[:1, :],
            self.creatures_with_border[-1:, :],
            self.creatures_with_border[1:-1, :1],
            self.creatures_with_border[1:-1, -1:],
        ]
        self.creature_storage = CreatureStorage() # creature info indexable by "creatures"

        self._food_rates = np.zeros((width, height), dtype=np.float64)
        self._cell_features: np.ndarray = None
        self._max_vision = 0
        self._cell_features_padding = -1 # negative value forces initialization of _cell_features
    
    def add_creature(self, genome: Genome) -> int:
        self._max_vision = max(self._max_vision, genome.vision_radius)
        index = self.creature_storage.allocate(genome.vision_radius)
        self.creature_storage.stats[index] = creature_stats.INITIAL_STATS
        self.creature_storage.param_coefficients[index] = genome.param_coefficients
        self.creature_storage.genomes[index] = genome
        feature_storage = self.creature_storage.features_storages[genome.vision_radius]
        features_index = self.creature_storage.features_indices[index]
        feature_storage.feature_coefficients[features_index] = genome.feature_coefficients()
        feature_storage.feature_biases[features_index] = genome.feature_biases()
        return index
    
    def setup_round(self) -> None:
        self._add_food_rate_spikes()
        self._decay_food_rates()
        self._add_food()
        self._abiogenesis()
    
    def cell_features(self) -> np.ndarray:
        if self._cell_features_padding < self._max_vision:
            self._cell_features_padding = self._max_vision
            self._cell_features = np.full((self.width, self.height, cell_stats.NUM_FEATURES), -1.0, dtype=np.float64)
        np.add(self.food, 1.0, out=self._cell_features[:, :, cell_stats.FOOD_FEATURE])
        np.exp(self._cell_features[:, :, cell_stats.FOOD_FEATURE], out=self._cell_features[:, :, cell_stats.FOOD_FEATURE])
        self._creature_public_features(out = self._cell_features[:, :, creature_stats.PUBLIC_LOG_FEATURES_START:creature_stats.PUBLIC_LOG_FEATURES_END])
        return self._cell_features
    
    def _add_food_rate_spikes(self) -> None:
        spikes = np.random.rand_like(self._food_rates)
        np.power(spikes, 50.0, out=spikes)
        np.add(self._food_rates, spikes, out=self._food_rates)

    def _add_food(self) -> None:
        np.add(self.food, self._food_rates, out=self.food)

    def _decay_food_rates(self) -> None:
        np.multiply(self._food_rates, 0.993, out=self._food_rates)
    
    _ABIOGENESIS_THRESHOLD = 100.0
    _ABIOGENESIS_CHANCE = 0.000001

    def _abiogenesis(self) -> None:
        spawn_mask = (self.food > self._ABIOGENESIS_THRESHOLD) & (np.random.rand_like(self.food) < self._ABIOGENESIS_CHANCE) & (self.creatures < 0)
        spawn_coords = np.where(spawn_mask)
        for y, x in zip(*spawn_coords):
            index = self.add_creature(Genome.random())
            self.creatures[y, x] = index
            self.food[y, x] -= self._ABIOGENESIS_THRESHOLD

    def _creature_public_features(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = np.empty((self.creature_storage.used_row_count(), creature_stats.NUM_PUBLIC_FEATURES), dtype=np.float64)
        indexed_features = creature.public_features(self.creature_storage)
        creature_present = self.creatures >= 0
        np.where(creature_present, indexed_features[out], creature.DEFAULT_PUBLIC_FEATURES, out=out)
        return out