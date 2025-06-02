import numpy as np
import cell_stats
import creature_stats
import network_outputs

def _feature_count(vision_radius: int) -> int:
    perception_feature_count = cell_stats.NUM_FEATURES * (2 * vision_radius + 1) ** 2
    private_feature_count = creature_stats.NUM_PRIVATE_FEATURES
    return perception_feature_count + private_feature_count

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
        self.stats = np.full((creature_stats.COUNT, 1), np.nan, dtype=np.float64)
        self.param_coefficients = np.full((network_outputs.PARAMS_COUNT, 1), np.nan, dtype=np.float64)
        
        # Track allocation state
        self._max_used_index = -1
        self._free_indices = set()
    
    def allocate(self) -> int:
        """Allocate a new row for a creature.
        
        Returns:
            int: Index of the allocated row
        """
        # Try to reuse a free index
        if self._free_indices:
            index = min(self._free_indices)
            self._free_indices.remove(index)
            return index
            
        # No free indices, use the next available index
        self._max_used_index += 1
        
        # If we've reached the end of the arrays, double their size
        if self._max_used_index >= self.row_count:
            new_row_count = self.row_count * 2
            # Create new arrays with doubled size
            new_stats = np.full((creature_stats.COUNT, new_row_count), np.nan, dtype=np.float64)
            new_param_coefficients = np.full((network_outputs.PARAMS_COUNT, new_row_count), np.nan, dtype=np.float64)
            # Copy old values
            new_stats[:, :self.row_count] = self.stats
            new_param_coefficients[:, :self.row_count] = self.param_coefficients
            # Replace old arrays
            self.stats = new_stats
            self.param_coefficients = new_param_coefficients
            self.row_count = new_row_count
            
        return self._max_used_index
    
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
            
        self._free_indices.add(index)
    
    def used_row_count(self) -> int:
        return self._max_used_index + 1