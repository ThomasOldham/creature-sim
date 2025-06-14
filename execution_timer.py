from collections import OrderedDict
from functools import wraps
import time
from typing import Dict, Optional

class ExecutionTimer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExecutionTimer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._indentation_levels: OrderedDict[str, int] = OrderedDict([
            ('Simulation.run_round', 0),
            ('Board.setup_round', 1),
            ('Board._add_food_rate_spikes', 2),
            ('Board._decay_food_rates', 2),
            ('Board._add_food', 2),
            ('Board._abiogenesis', 2),
            ('Genome.__init__', 3),
            ('MutationControl.__init__', 4),
            ('Board.add_creature', 3),
            ('CreatureStorage.allocate', 4),
            ('InputTransformStorage.__init__', 5),
            ('InputTransformStorage.allocate', 5),
            ('Genome.brain_mass', 4),
            ('Genome.feature_coefficients', 4),
            ('Genome.feature_biases', 4),
            ('creature.recalculate_min_mass', 2),
            ('Simulation._creature_turns', 1),
            ('Simulation._decide_actions', 2),
            ('Simulation._get_outputs', 3),
            ('Board.all_features', 4),
            ('creature.private_features', 5),
            ('Board._get_cell_features', 5),
            ('Board._creature_public_features', 6),
            ('creature.public_features', 7),
            ('Board._untransformed_features_for_vision_radius', 5),
            ('creature.transform_features', 5),
            ('creature.action_features', 4),
            ('NeuralNetwork.forward', 4),
            ('creature.decide_action_kind', 3),
            ('creature._normalize_action_probabilities', 4),
            ('creature.action_params', 3),
            ('creature._snap_direction', 4),
            ('creature._available_mass', 4),
            ('creature.reset_short_term_memory', 2),
            ('Simulation._execute_actions', 2),
            ('action.attack_action', 3),
            ('Board.check_and_apply_killings', 4),
            ('action.eat_action', 3),
            ('action.heal_action', 3),
            ('action.upgrade_action', 3),
            ('action.move_action', 3),
            ('action._handle_grid_movement', 4),
            ('action.reproduce_action', 3),
            ('action._fidelity_tax', 4),
            ('action._select_actor_per_target', 4),
            ('Genome.size', 5),
            ('Simulation._apply_action_results', 2),
            ('Board.wrapup_round', 1),
            ('Board._apply_bmr', 2),
            ('Board._apply_starvation', 2),
            ('Board.apply_death', 3),
            ('CreatureStorage.release', 4),
            ('InputTransformStorage.release', 5),
        ])
        self._timers: Dict[str, float] = {
            method: 0.0 for method in self._indentation_levels.keys()
        }
    
    def get_time(self, method_name: str) -> float:
        """Get the total time spent in a specific method.
        
        Args:
            method_name: The name of the method to get timing for
            
        Returns:
            float: Total time spent in the method in seconds
        """
        return self._timers[method_name]
    
    def add_time(self, method_name: str, elapsed_time: float) -> None:
        """Add time to a method's total.
        
        Args:
            method_name: The name of the method to add time to
            elapsed_time: The time to add in seconds
        """
        self._timers[method_name] += elapsed_time
    
    def reset(self, method_name: Optional[str] = None) -> None:
        """Reset timing for a specific method or all methods.
        
        Args:
            method_name: The name of the method to reset, or None to reset all
        """
        if method_name is None:
            self._initialize()
        elif method_name in self._timers:
            self._timers[method_name] = 0.0
    
    def to_formatted_string(self) -> str:
        """Get a formatted string showing all timing information.
        
        Returns:
            str: Multi-line string with timing information
        """
        lines = ["Execution Timer Statistics:"]
        for method, indentation_level in self._indentation_levels.items():
            lines.append("\t"*indentation_level + f"{method}: {self._timers[method]:.6f} seconds")
        return "\n".join(lines)

def timer_decorator(method_name: str):
    """Decorator to time method execution.
    
    Args:
        method_name: The name to use for timing this method
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = ExecutionTimer()
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            timer.add_time(method_name, elapsed_time)
            return result
        return wrapper
    return decorator 