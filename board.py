from typing import List, Optional
from creature_storage import CreatureStorage
from genome import Genome
import cell_stats
import creature
import creature_stats
import numpy as np
from execution_timer import timer_decorator

class Board:
    def __init__(self, width: int, height: int):
        self.width = width # set at construction and fixed for the lifetime of the board
        self.height = height # set at construction and fixed for the lifetime of the board
        self.food = np.zeros((height, width), dtype=np.float64) # food amounts on the board (should never be negative)
        self.creatures_with_border = np.full((height + 2, width + 2), -1.0, dtype=np.int64) # view of creatures with a border manipulable through border_pseudocreatures
        self.creatures = self.creatures_with_border[1:-1, 1:-1] # indices of creatures on the board
        self.border_pseudocreatures = [ # mutable interface for arbitrarily setting border conditions to help handle edge cases
            self.creatures_with_border[:1, :],
            self.creatures_with_border[-1:, :],
            self.creatures_with_border[1:-1, :1],
            self.creatures_with_border[1:-1, -1:],
        ]
        self.creature_storage = CreatureStorage() # creature info indexable by "creatures"

        self._food_rates = np.zeros((height, width), dtype=np.float64)
        self._cell_features: np.ndarray = None
        self._max_vision = 1
        self._cell_features_padding = -1 # negative value forces initialization of _cell_features
    
    @timer_decorator('Board.add_creature')
    def add_creature(self, genome: Genome) -> int:
        self._max_vision = max(self._max_vision, genome.vision_radius)
        index = self.creature_storage.allocate(genome.vision_radius)
        self.creature_storage.stats[index] = creature_stats.STARTING_VALUES
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
    
    @timer_decorator('Board.setup_round')
    def setup_round(self) -> None:
        self._add_food_rate_spikes()
        self._decay_food_rates()
        self._add_food()
        self._abiogenesis()
        creature.recalculate_min_mass(self.creature_storage)
    
    @timer_decorator('Board.wrapup_round')
    def wrapup_round(self) -> None:
        self._apply_bmr()
        self._apply_starvation()

    @timer_decorator('Board.apply_death')
    def apply_death(self, dead_creature_indices: np.ndarray) -> None:
        self.creature_storage.is_alive[dead_creature_indices] = False
        dead_positions = self.creature_storage.grid_position[dead_creature_indices]
        dead_masses = np.maximum(self.creature_storage.stats[:, creature_stats.MIN_MASS],
                                  self.creature_storage.stats[:, creature_stats.MASS])
        self.food[dead_positions[:, 1], dead_positions[:, 0]] += dead_masses
        self.creatures[dead_positions[:, 1], dead_positions[:, 0]] = -1
        self.creature_storage.is_alive[dead_creature_indices] = False
        for index in dead_creature_indices:
            self.creature_storage.release(index)
    
    @timer_decorator('Board.all_features')
    def transformed_features(self, outs: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        creature_storage = self.creature_storage
        if not outs:
            outs = [np.zeros((storage.used_row_count(), storage.feature_count)) for storage in creature_storage.features_storages]
        all_private_features = creature.private_features(creature_storage)
        all_cell_features = self._get_cell_features()
        for vision_radius in range(self._max_vision):
            self._untransformed_features_for_vision_radius(vision_radius, all_private_features, all_cell_features, outs[vision_radius])
        creature.transform_features(outs, creature_storage)
        return outs
    
    @timer_decorator('Board._add_food_rate_spikes')
    def _add_food_rate_spikes(self) -> None:
        spikes = np.random.rand(self.height, self.width)
        np.power(spikes, 50.0, out=spikes)
        np.add(self._food_rates, spikes, out=self._food_rates)

    @timer_decorator('Board._add_food')
    def _add_food(self) -> None:
        np.add(self.food, self._food_rates, out=self.food)

    @timer_decorator('Board._decay_food_rates')
    def _decay_food_rates(self) -> None:
        np.multiply(self._food_rates, 0.993, out=self._food_rates)
    
    _ABIOGENESIS_THRESHOLD = 100.0
    _ABIOGENESIS_CHANCE = 0.000001

    @timer_decorator('Board._abiogenesis')
    def _abiogenesis(self) -> None:
        spawn_mask = (self.food > self._ABIOGENESIS_THRESHOLD) & (np.random.rand(self.height, self.width) < self._ABIOGENESIS_CHANCE) & (self.creatures < 0)
        spawn_coords = np.where(spawn_mask)
        for y, x in zip(*spawn_coords):
            index = self.add_creature(Genome())
            self.creatures[y, x] = index
            self.food[y, x] -= self._ABIOGENESIS_THRESHOLD
            self.creature_storage.grid_position[index] = [x, y]
            self.creature_storage.stats[index, creature_stats.MASS] = self._ABIOGENESIS_THRESHOLD
    
    @timer_decorator('Board._get_cell_features')
    def _get_cell_features(self) -> np.ndarray:
        if self._cell_features_padding < self._max_vision:
            self._cell_features_padding = self._max_vision
            self._cell_features = np.full((self.height+(2*self._max_vision), self.width+(2*self._max_vision), cell_stats.NUM_FEATURES), -1.0, dtype=np.float64)
        np.add(self.food, 1.0, out=self._cell_features[self._max_vision:-self._max_vision, self._max_vision:-self._max_vision, cell_stats.FOOD_FEATURE])
        np.exp(self._cell_features[self._max_vision:-self._max_vision, self._max_vision:-self._max_vision, :cell_stats.NUM_GROUND_FEATURES], out=self._cell_features[self._max_vision:-self._max_vision, self._max_vision:-self._max_vision, :cell_stats.NUM_GROUND_FEATURES])
        self._creature_public_features(out = self._cell_features[self._max_vision:-self._max_vision, self._max_vision:-self._max_vision, cell_stats.NUM_GROUND_FEATURES:])
        return self._cell_features

    @timer_decorator('Board._creature_public_features')
    def _creature_public_features(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = np.empty((self.creature_storage.used_row_count(), creature_stats.NUM_PUBLIC_FEATURES), dtype=np.float64)
        indexed_features = creature.public_features(self.creature_storage)
        creature_present = self.creatures >= 0
        out[:] = creature.DEFAULT_PUBLIC_FEATURES
        out[creature_present] = indexed_features[self.creatures[creature_present]]
        return out
    
    @timer_decorator('Board._untransformed_features_for_vision_radius')
    def _untransformed_features_for_vision_radius(self, vision_radius: int, all_private_features: np.ndarray, all_cell_features: np.ndarray, out: np.ndarray) -> None:
        creature_storage = self.creature_storage
        features_storage = creature_storage.features_storages[vision_radius]
        # TODO: Actually supporting multiple vision radii is hard, so use just one radius and
        #       skip the rest for now. This means we assume only one features_storage is populated,
        #       and it's parallel to the creatures_storage.
        if features_storage._max_used_index < 0:
            return
        
        # self features
        self_features = all_private_features
        out[:, :creature_stats.NUM_PRIVATE_FEATURES] = self_features

        # perception features
        creature_coords = creature_storage.grid_position
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
        out[:, creature_stats.NUM_PRIVATE_FEATURES:] = perception_features
    
    @timer_decorator('Board._apply_bmr')
    def _apply_bmr(self) -> None:
        bmr = self.creature_storage.stats[:, creature_stats.MASS] + self.creature_storage.stats[:, creature_stats.MIN_MASS]
        np.divide(bmr, 2000.0, out=bmr)
        np.subtract(self.creature_storage.stats[:, creature_stats.MASS], bmr, out=self.creature_storage.stats[:, creature_stats.MASS])

    @timer_decorator('Board._apply_starvation')
    def _apply_starvation(self) -> None:
        stats = self.creature_storage.stats[:self.creature_storage.used_row_count()]
        starved_mask = stats[:, creature_stats.MASS] < stats[:, creature_stats.MIN_MASS]
        self.apply_death(np.where(starved_mask)[0])

    @timer_decorator('Board.check_and_apply_killings')
    def check_and_apply_killings(self) -> None:
        """Kills any creature whose damage is greater than or equal to its max HP."""
        stats = self.creature_storage.stats[:self.creature_storage.used_row_count()]
        killed_mask = stats[:, creature_stats.DAMAGE] >= stats[:, creature_stats.MAX_HP]
        self.apply_death(np.where(killed_mask)[0])

    def display_ascii(self) -> None:
        """Create and print an ASCII representation of the board.
        
        Each square shows:
        - '.' for empty squares (with or without food)
        - 'C' for a creature
        
        The board is shown with row and column numbers for clarity.
        """
        # If width>10, add column 10s-place numbers at the top
        if self.width > 10:
            header = "   "  # Space for row numbers
            for x in range(self.width):
                header += f"{x // 10}" if x % 10 == 0 else " "
            print(header)

        # Add column numbers at the top
        header = "   "  # Space for row numbers
        for x in range(self.width):
            header += f"{x % 10}"  # Use modulo 10 to keep single digits
        print(header)
        
        # Add a separator line
        print("   " + "-" * self.width)
        
        # Add each row with its row number
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.creatures[y, x] >= 0:
                    row.append('C')
                else:
                    row.append('.')
            # Add row number and content
            print(f"{y:2d}|{''.join(row)}")