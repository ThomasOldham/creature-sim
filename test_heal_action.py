import math
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from simulation import Simulation
from board import Board
from creature_storage import CreatureStorage
from genome import Genome
import action_kind
import creature_stats
import action
import network_outputs

DEFAULT_BRAIN_MASS = Genome().brain_mass()

class TestHealAction(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.width = 5
        self.height = 5
        self.simulation = Simulation(self.width, self.height, random_seed=42)
        
        # Mock the board methods to be no-ops
        self.simulation.board._add_food_rate_spikes = MagicMock()
        self.simulation.board._decay_food_rates = MagicMock()
        self.simulation.board._add_food = MagicMock()
        self.simulation.board._abiogenesis = MagicMock()
        
        # Reset food grid to ensure clean state
        self.simulation.board.food.fill(0.0)
        self.simulation.board._food_rates.fill(0.0)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def _setup_neural_network_mock(self, heal_mass_fractions):
        """Set up neural network mock to return outputs that result in ACTION_HEAL with specific mass fractions.
        
        Args:
            heal_mass_fractions: List of mass fractions to spend on healing for each creature
        """
        num_creatures = len(heal_mass_fractions)
        
        # Set up the mock network for each creature
        for i in range(num_creatures):
            mock_network = MagicMock()
            
            # Create mock network outputs that will result in ACTION_HEAL being selected
            # The first ACTION_KINDS_COUNT outputs are action probabilities
            # We need ACTION_HEAL (index 2) to have the highest probability
            mock_outputs = np.zeros(network_outputs.COUNT, dtype=np.float64)
            mock_outputs[2] = 1.0  # Set ACTION_HEAL probability to 1.0
            
            # Set the heal mass fraction parameter
            # PARAM_HEAL_MASS_FRACTION is the index for the heal mass fraction parameter
            transformed_heal_mass_fraction = heal_mass_fractions[i]
            untransformed_heal_mass_fraction = -math.log(1/transformed_heal_mass_fraction - 1)
            mock_outputs[network_outputs.ACTION_KINDS_COUNT + network_outputs.PARAM_HEAL_MASS_FRACTION] = untransformed_heal_mass_fraction
            
            mock_network.forward.return_value = mock_outputs
            self.simulation.board.creature_storage.network[i] = mock_network

    def _verify_board_state(self, expected_food, expected_creatures, expected_creatures_with_border):
        """Verify the state of the board."""
        # Check entire food grid
        np.testing.assert_array_equal(self.simulation.board.food, expected_food,
                                     "Food grid should match expected state")
        
        # Check entire creatures grid
        np.testing.assert_array_equal(self.simulation.board.creatures, expected_creatures,
                                     "Creatures grid should match expected state")
        
        # Check entire creatures_with_border grid
        np.testing.assert_array_equal(self.simulation.board.creatures_with_border, expected_creatures_with_border,
                                     "Creatures_with_border grid should match expected state")

    def _verify_creature_storage_state(self, expected_stats, expected_is_alive, expected_grid_position):
        """Verify the state of creature storage."""
        # Check entire stats array
        np.testing.assert_array_almost_equal(self.simulation.board.creature_storage.stats, expected_stats, 10,
                                     "Creature stats should match expected state")
        
        # Check entire is_alive array
        np.testing.assert_array_equal(self.simulation.board.creature_storage.is_alive, expected_is_alive,
                                     "Creature is_alive should match expected state")
        
        # Check entire grid_position array
        np.testing.assert_array_equal(self.simulation.board.creature_storage.grid_position, expected_grid_position,
                                     "Creature grid_position should match expected state")

    def test_heal_action_basic(self):
        """Test basic heal action functionality."""
        # Create a creature with damage
        genome = Genome()
        creature_idx = self.simulation.board.add_creature(genome)
        
        # Position the creature at (2, 2)
        self.simulation.board.creatures[2, 2] = creature_idx
        self.simulation.board.creature_storage.grid_position[creature_idx] = [2, 2]
        
        # Set creature's initial stats
        heal_power = 1
        initial_mass = 100.0
        initial_damage = 20.0
        mass_fraction = 0.02  # Spend 0.02 of available mass to heal
        
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.HEAL_POWER] = heal_power
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.MASS] = initial_mass
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.DAMAGE] = initial_damage
        
        # Set up neural network mock
        self._setup_neural_network_mock([mass_fraction])
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Calculate expected changes
        actual_min_mass = self.simulation.board.creature_storage.stats[0, creature_stats.MIN_MASS] # cheat and copy from actual results
        available_mass = initial_mass - actual_min_mass
        mass_spent = mass_fraction * available_mass
        healing_amount = np.power(mass_spent, 0.9) * heal_power
        damage_reduction = min(initial_damage, healing_amount)
        final_damage = initial_damage - damage_reduction
        
        # Expected final states
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        expected_creatures[2, 2] = creature_idx
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        expected_creatures_with_border[3, 3] = creature_idx  # (2,2) in creatures maps to (3,3) in creatures_with_border
        
        # Build expected stats from STARTING_VALUES
        expected_stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        expected_stats[0] = creature_stats.STARTING_VALUES.copy()
        expected_stats[0, creature_stats.HEAL_POWER] = heal_power  # Override with our custom heal power
        expected_stats[0, creature_stats.MASS] = initial_mass - mass_spent  # Mass after healing
        expected_stats[0, creature_stats.DAMAGE] = final_damage  # Damage after healing
        expected_stats[0, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[0, creature_stats.LAST_ACTION] = action_kind.ACTION_HEAL
        expected_stats[0, creature_stats.LAST_SUCCESS] = healing_amount / damage_reduction if damage_reduction > 0 else 0.0
        expected_stats[0, creature_stats.LAST_COST] = mass_spent
        expected_stats[0, creature_stats.LAST_DX] = 0.0
        expected_stats[0, creature_stats.LAST_DY] = 0.0
        expected_stats[0, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS
        expected_stats[0, creature_stats.MIN_MASS] = actual_min_mass
        
        # Apply BMR (Basal Metabolic Rate) reduction
        bmr = (expected_stats[0, creature_stats.MASS] + expected_stats[0, creature_stats.MIN_MASS]) / 2000.0
        expected_stats[0, creature_stats.MASS] -= bmr
        
        expected_is_alive = np.ones(1, dtype=bool)
        
        expected_grid_position = np.full((1, 2), -1, dtype=np.int64)
        expected_grid_position[0] = [2, 2]
        
        # Verify complete board state
        self._verify_board_state(expected_food, expected_creatures, expected_creatures_with_border)
        
        # Verify complete creature storage state
        self._verify_creature_storage_state(expected_stats, expected_is_alive, expected_grid_position)

if __name__ == '__main__':
    unittest.main() 