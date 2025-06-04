from typing import List
import numpy as np
import creature_stats
from genome import Genome
import network_outputs
from input_transform_storage import InputTransformStorage
from neural_network import NeuralNetwork
from execution_timer import timer_decorator

class CreatureStorage:
    """Manages parallel arrays collectively representing Creatures.
    
    This class maintains arrays of creature stats and parameters, with each creature
    occupying a row in these arrays. The arrays grow dynamically as needed, starting
    with one row and doubling in size when full.
    """
    
    def __init__(self):
        """Initialize the creature storage with parallel arrays starting at size 1."""
        self.row_count = 1
        
        # Initialize parallel arrays with one row
        self._stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        self._param_coefficients = np.full((1, network_outputs.PARAMS_COUNT), np.nan, dtype=np.float64)
        self._is_alive = np.zeros(1, dtype=bool)
        self._genome = np.full(1, None, dtype=object)
        self._network = np.full(1, None, dtype=object)
        self._grid_position = np.full((1, 2), -1, dtype=np.int64)
        
        # Track allocation state
        self._max_used_index = -1
        self._free_indices = set()
        
        # Vision radius tracking
        self._vision_radius = np.full(1, -1, dtype=np.int64)
        self._features_index = np.full(1, -1, dtype=np.int64)
        self.features_storages = [InputTransformStorage(0), InputTransformStorage(1)]
        
    @timer_decorator('CreatureStorage.allocate')
    def allocate(self, creature_vision_radius: int) -> int:
        """Allocate a new row for a creature.
        
        Args:
            creature_vision_radius: The vision radius for the new creature
            
        Returns:
            int: Index of the allocated row
        """
        # Try to reuse a free index
        if self._free_indices:
            index = self._free_indices.pop()
        else:
            # No free indices, use the next available index
            self._max_used_index += 1
            index = self._max_used_index
            
            # If we've reached the end of the arrays, double their size
            if index >= self.row_count:
                new_row_count = self.row_count * 2
                # Create new arrays with doubled size
                new_stats = np.full((new_row_count, creature_stats.COUNT), np.nan, dtype=np.float64)
                new_param_coefficients = np.full((new_row_count, network_outputs.PARAMS_COUNT), np.nan, dtype=np.float64)
                new_is_alive = np.zeros(new_row_count, dtype=bool)
                new_vision_radius = np.full(new_row_count, -1, dtype=np.int64)
                new_features_index = np.full(new_row_count, -1, dtype=np.int64)
                new_genome = np.full(new_row_count, None, dtype=object)
                new_network = np.full(new_row_count, None, dtype=object)
                new_grid_position = np.full((new_row_count, 2), -1, dtype=np.int64)
                # Copy old values
                new_stats[:self.row_count, :] = self._stats
                new_param_coefficients[:self.row_count, :] = self._param_coefficients
                new_is_alive[:self.row_count] = self._is_alive
                new_vision_radius[:self.row_count] = self._vision_radius
                new_features_index[:self.row_count] = self._features_index
                new_genome[:self.row_count] = self._genome
                new_network[:self.row_count] = self._network
                new_grid_position[:self.row_count, :] = self._grid_position
                # Replace old arrays
                self._stats = new_stats
                self._param_coefficients = new_param_coefficients
                self._is_alive = new_is_alive
                self._vision_radius = new_vision_radius
                self._features_index = new_features_index
                self._genome = new_genome
                self._network = new_network
                self._grid_position = new_grid_position
                self.row_count = new_row_count
        
        # Ensure we have enough features storages
        while creature_vision_radius >= len(self.features_storages):
            self.features_storages.append(InputTransformStorage(len(self.features_storages)))
            
        # Allocate in the appropriate features storage
        features_index = self.features_storages[creature_vision_radius].allocate()
        
        # Record the vision radius and features index
        self._vision_radius[index] = creature_vision_radius
        self._features_index[index] = features_index
        
        return index
    
    @timer_decorator('CreatureStorage.release')
    def release(self, index: int) -> None:
        """Release a row back to the pool.
        
        Args:
            index: Index of the row to release
            
        Raises:
            ValueError: If the index is invalid or already released
        """
        if not 0 <= index <= self._max_used_index:
            raise ValueError(f"Invalid row index: {index}")
        if index in self._free_indices:
            raise ValueError(f"Row {index} is already released")
            
        # Release the features storage allocation
        creature_vision_radius = self._vision_radius[index]
        features_index = self._features_index[index]
        self.features_storages[creature_vision_radius].release(features_index)
        self._free_indices.add(index)
    
    def used_row_count(self) -> int:
        """Get the number of used rows in the storage.
        
        Returns:
            int: The number of used rows
        """
        return self._max_used_index + 1
    
    @property
    def stats(self) -> np.ndarray:
        return self._stats[:self.used_row_count()]
    
    @property
    def param_coefficients(self) -> np.ndarray:
        return self._param_coefficients[:self.used_row_count()]
    
    @property
    def is_alive(self) -> np.ndarray:
        return self._is_alive[:self.used_row_count()]
    
    @property
    def genome(self) -> np.ndarray:
        return self._genome[:self.used_row_count()]
    
    @property
    def network(self) -> np.ndarray:
        return self._network[:self.used_row_count()]
    
    @property
    def grid_position(self) -> np.ndarray:
        return self._grid_position[:self.used_row_count()]
    
    @property
    def vision_radius(self) -> np.ndarray:
        return self._vision_radius[:self.used_row_count()]
    
    @property
    def features_index(self) -> np.ndarray:
        return self._features_index[:self.used_row_count()]