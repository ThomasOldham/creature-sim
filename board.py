from typing import List, Optional
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
        self.creature_storage.stats[index, creature_stats.BRAIN_MASS] = genome.brain_mass()
        self.creature_storage.param_coefficients[index] = genome.param_coefficients
        self.creature_storage.genome[index] = genome
        self.creature_storage.network[index] = genome.network
        self.creature_storage.is_alive[index] = True
        feature_storage = self.creature_storage.features_storages[genome.vision_radius]
        features_index = self.creature_storage.features_index[index]
        feature_storage.feature_coefficients[features_index] = genome.feature_coefficients()
        feature_storage.feature_biases[features_index] = genome.feature_biases()
        return index
    
    def setup_round(self) -> None:
        self._add_food_rate_spikes()
        self._decay_food_rates()
        self._add_food()
        self._abiogenesis()
        creature.recalculate_min_mass(self.creature_storage)
    
    def wrapup_round(self) -> None:
        self._apply_bmr()
        self._apply_starvation()

    def apply_death(self, dead_creature_indices: np.ndarray) -> None:
        self.creature_storage.is_alive[dead_creature_indices] = False
        dead_positions = self.creature_storage.grid_position[dead_creature_indices]
        dead_masses = np.maximum(self.creature_storage.stats[:, creature_stats.MIN_MASS],
                                  self.creature_storage.stats[:, creature_stats.MASS])
        self.food[dead_positions[:, 1], dead_positions[:, 0]] += dead_masses
        self.creatures[dead_positions[:, 1], dead_positions[:, 0]] = -1
        self.creature_storage.is_alive[dead_creature_indices] = False
    
    def all_features(self, outs: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        creature_storage = self.creature_storage
        if not outs:
            outs = [np.zeros((storage.used_row_count(), storage.feature_count)) for storage in creature_storage.features_storages]
        all_private_features = creature.private_features(creature_storage)
        all_cell_features = self._get_cell_features()
        for vision_radius in range(self._max_vision):
            self._untransformed_features_for_vision_radius(vision_radius, all_private_features, all_cell_features, outs[vision_radius])
        creature.transform_features(outs, creature_storage)
        return outs
    
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
            self.creature_storage.grid_position[index] = np.array([x, y], dtype=np.int64)
            self.creature_storage.stats[index, creature_stats.MASS] = self._ABIOGENESIS_THRESHOLD
    
    def _get_cell_features(self) -> np.ndarray:
        if self._cell_features_padding < self._max_vision:
            self._cell_features_padding = self._max_vision
            self._cell_features = np.full((self.width, self.height, cell_stats.NUM_FEATURES), -1.0, dtype=np.float64)
        np.add(self.food, 1.0, out=self._cell_features[:, :, cell_stats.FOOD_FEATURE])
        np.exp(self._cell_features[:, :, cell_stats.FOOD_FEATURE], out=self._cell_features[:, :, cell_stats.FOOD_FEATURE])
        self._creature_public_features(out = self._cell_features[:, :, creature_stats.PUBLIC_LOG_FEATURES_START:creature_stats.PUBLIC_LOG_FEATURES_END])
        return self._cell_features

    def _creature_public_features(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = np.empty((self.creature_storage.used_row_count(), creature_stats.NUM_PUBLIC_FEATURES), dtype=np.float64)
        indexed_features = creature.public_features(self.creature_storage)
        creature_present = self.creatures >= 0
        np.where(creature_present, indexed_features[out], creature.DEFAULT_PUBLIC_FEATURES, out=out)
        return out
    
    def _untransformed_features_for_vision_radius(self, vision_radius: int, all_private_features: np.ndarray, all_cell_features: np.ndarray, out: np.ndarray) -> None:
        creature_storage = self.creature_storage
        features_storage = creature_storage.features_storages[vision_radius]
        # TODO: Actually supporting multiple vision radii is hard, so use just one radius and
        #       skip the rest for now. This means we assume only one features_storage is populated,
        #       and it's parallel to the creatures_storage.
        if features_storage.max_used_index() == 0:
            return
        
        # self features
        self_features = all_private_features
        out[:creature_stats.NUM_PRIVATE_FEATURES] = self_features

        # perception features
        creature_coords = creature_storage.grid_position[creature_storage.used_row_count()]
        grid_x = creature_coords[:, 0]
        grid_y = creature_coords[:, 1]
        vision_length = 2*vision_radius+1
        perceived_x = np.empty((grid_x.shape[0], vision_length, vision_length), dtype=np.int64)
        perceived_x[:] = np.arange(vision_length)[:]
        perceived_x += grid_x[:, np.newaxis, np.newaxis]
        perceived_y = np.empty((grid_x.shape[0], vision_length, vision_length), dtype=np.int64)
        perceived_y[:] = np.arange(vision_length)[:, np.newaxis]
        perceived_y += grid_y[:, np.newaxis, np.newaxis]
        perception_squares = all_cell_features[perceived_y, perceived_x]
        perception_features = perception_squares.reshape(len(grid_x), -1)
        out[creature_stats.NUM_PRIVATE_FEATURES:] = perception_features
    
    def _apply_bmr(self) -> None:
        bmr = self.creature_storage.stats[:, creature_stats.MASS] + self.creature_storage.stats[:, creature_stats.MIN_MASS]
        np.divide(bmr, 2000.0, out=bmr)
        np.subtract(self.creature_storage.stats[:, creature_stats.MASS], bmr, out=self.creature_storage.stats[:, creature_stats.MASS])

    def _apply_starvation(self) -> None:
        starved_mask = self.creature_storage.stats[:, creature_stats.MASS] < self.creature_storage.stats[:, creature_stats.MIN_MASS]
        self.apply_death(np.where(starved_mask))