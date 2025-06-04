import numpy as np
import cell_stats
import creature_stats
from execution_timer import timer_decorator

def num_features_for_vision(vision_radius: int) -> int:
    """Calculate the number of features for a creature with the given vision radius.
    
    Args:
        vision_radius: The creature's vision radius
        
    Returns:
        int: Total number of features (self features + perception features)
    """
    num_visible_squares = (2 * vision_radius + 1) * (2 * vision_radius + 1)
    num_perception_features = cell_stats.NUM_FEATURES
    return creature_stats.NUM_PRIVATE_FEATURES + num_visible_squares * num_perception_features

class InputTransformStorage:
    """Manages parallel arrays for feature coefficients and biases for creatures with a specific vision radius.
    
    This class maintains arrays of feature coefficients and biases, with each creature
    occupying a row in these arrays. The arrays grow dynamically as needed, starting
    with one row and doubling in size when full.
    """
    
    @timer_decorator('InputTransformStorage.__init__')
    def __init__(self, vision_radius: int):
        """Initialize the input transform storage with parallel arrays starting at size 1.
        
        Args:
            vision_radius: The vision radius for all creatures in this storage
        """
        self.feature_count = num_features_for_vision(vision_radius)
        self.row_count = 1
        
        # Initialize parallel arrays with one row
        self._feature_coefficients = np.full((1, self.feature_count), np.nan, dtype=np.float64)
        self._feature_biases = np.full((1, self.feature_count), np.nan, dtype=np.float64)
        
        # Track allocation state
        self._max_used_index = -1
        self._free_indices = set()
    
    @timer_decorator('InputTransformStorage.allocate')
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
            new_feature_coefficients = np.full((new_row_count, self.feature_count), np.nan, dtype=np.float64)
            new_feature_biases = np.full((new_row_count, self.feature_count), np.nan, dtype=np.float64)
            # Copy old values
            new_feature_coefficients[:self.row_count, :] = self._feature_coefficients
            new_feature_biases[:self.row_count, :] = self._feature_biases
            # Replace old arrays
            self._feature_coefficients = new_feature_coefficients
            self._feature_biases = new_feature_biases
            self.row_count = new_row_count
            
        return self._max_used_index
    
    @timer_decorator('InputTransformStorage.release')
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
        """Get the number of rows currently in use.
        
        Returns:
            int: Number of rows in use
        """
        return self._max_used_index + 1
    
    @property
    def feature_coefficients(self) -> np.ndarray:
        return self._feature_coefficients[:self.used_row_count()]
    
    @property
    def feature_biases(self) -> np.ndarray:
        return self._feature_biases[:self.used_row_count()]