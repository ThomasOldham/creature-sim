import numpy as np
import creature_stats
import network_outputs
from input_transform_storage import InputTransformStorage

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
        self.stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        self.param_coefficients = np.full((1, network_outputs.PARAMS_COUNT), np.nan, dtype=np.float64)
        self.is_alive = np.zeros(1, dtype=bool)
        self.genome = [None]
        self.grid_position = np.full((1, 2), -1, dtype=np.int64)
        
        # Track allocation state
        self._max_used_index = -1
        self._free_indices = set()
        
        # Vision radius tracking
        self.vision_radius = np.full(1, -1, dtype=np.int64)
        self.features_index = np.full(1, -1, dtype=np.int64)
        self.features_storages = []
        
    def allocate(self, vision_radius: int) -> int:
        """Allocate a new row for a creature.
        
        Args:
            vision_radius: The vision radius for this creature
            
        Returns:
            int: Index of the allocated row
        """
        # Try to reuse a free index
        if self._free_indices:
            index = min(self._free_indices)
            self._free_indices.remove(index)
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
                new_genome = [None] * new_row_count
                new_grid_position = np.full((new_row_count, 2), -1, dtype=np.int64)
                # Copy old values
                new_stats[:self.row_count, :] = self.stats
                new_param_coefficients[:self.row_count, :] = self.param_coefficients
                new_is_alive[:self.row_count] = self.is_alive
                new_vision_radius[:self.row_count] = self.vision_radius
                new_features_index[:self.row_count] = self.features_index
                new_genome[:self.row_count] = self.genome
                new_grid_position[:self.row_count, :] = self.grid_position
                # Replace old arrays
                self.stats = new_stats
                self.param_coefficients = new_param_coefficients
                self.is_alive = new_is_alive
                self.vision_radius = new_vision_radius
                self.features_index = new_features_index
                self.genome = new_genome
                self.grid_position = new_grid_position
                self.row_count = new_row_count
        
        # Ensure we have enough features storages
        while vision_radius >= len(self.features_storages):
            self.features_storages.append(InputTransformStorage(len(self.features_storages)))
            
        # Allocate in the appropriate features storage
        features_index = self.features_storages[vision_radius].allocate()
        
        # Record the vision radius and features index
        self.vision_radius[index] = vision_radius
        self.features_index[index] = features_index
        
        return index
    
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
        vision_radius = self.vision_radius[index]
        features_index = self.features_index[index]
        self.features_storages[vision_radius].release(features_index)
        self._free_indices.add(index)
    
    def used_row_count(self) -> int:
        """Get the number of rows currently in use.
        
        Returns:
            int: Number of rows in use
        """
        return self._max_used_index + 1