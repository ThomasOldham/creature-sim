from typing import List
import numpy as np

class CreatureDebugInfo:
    """Debug information for a creature, including its ID and ancestry."""
    
    def __init__(self):
        """Initialize a new CreatureDebugInfo with a random ID and empty ancestors list."""
        self.id = np.random.randint(0, 2**32)
        self.ancestors: List[int] = [] 