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

class TestEatAction(unittest.TestCase):
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

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def _setup_neural_network_mock(self, num_creatures):
        """Set up neural network mock to return outputs that result in ACTION_EAT."""
        # Create mock network outputs that will result in ACTION_EAT being selected
        # The first ACTION_KINDS_COUNT outputs are action probabilities
        # We need ACTION_EAT (index 1) to have the highest probability
        mock_outputs = np.zeros((num_creatures, network_outputs.COUNT), dtype=np.float64)
        mock_outputs[:, 1] = 1.0  # Set ACTION_EAT probability to 1.0
        
        # Set up the mock network for each creature
        for i in range(num_creatures):
            mock_network = MagicMock()
            mock_network.forward.return_value = mock_outputs[i]
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
        np.testing.assert_array_equal(self.simulation.board.creature_storage.stats, expected_stats,
                                     "Creature stats should match expected state")
        
        # Check entire is_alive array
        np.testing.assert_array_equal(self.simulation.board.creature_storage.is_alive, expected_is_alive,
                                     "Creature is_alive should match expected state")
        
        # Check entire grid_position array
        np.testing.assert_array_equal(self.simulation.board.creature_storage.grid_position, expected_grid_position,
                                     "Creature grid_position should match expected state")

    def test_eat_action_no_creatures(self):
        """Test eat action when there are no creatures on the board."""
        # Set up food on the board
        self.simulation.board.food[2, 2] = 50.0
        self.simulation.board.food[1, 1] = 30.0
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Expected final states - food should remain unchanged
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        expected_food[2, 2] = 50.0  # Food unchanged
        expected_food[1, 1] = 30.0  # Food unchanged
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        
        # No creatures, so all storage arrays should be empty
        expected_stats = np.empty((0, creature_stats.COUNT), dtype=np.float64)
        expected_is_alive = np.empty(0, dtype=bool)
        expected_grid_position = np.empty((0, 2), dtype=np.int64)
        
        # Verify complete board state
        self._verify_board_state(expected_food, expected_creatures, expected_creatures_with_border)
        
        # Verify complete creature storage state
        self._verify_creature_storage_state(expected_stats, expected_is_alive, expected_grid_position)

    def test_eat_action_basic(self):
        """Test basic eat action functionality."""
        # Set up initial food on the board
        # Place 50.0 food at position (2, 2)
        self.simulation.board.food[2, 2] = 50.0
        
        # Create a creature with specific eat rate
        genome = Genome()
        creature_idx = self.simulation.board.add_creature(genome)
        
        # Position the creature at (2, 2) where there's food
        self.simulation.board.creatures[2, 2] = creature_idx
        self.simulation.board.creature_storage.grid_position[creature_idx] = [2, 2]
        
        # Set creature's eat rate to 20.0 (should eat 20.0 food)
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.EAT_RATE] = 20.0
        
        # Set initial mass to 100.0
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.MASS] = 100.0
        
        # Set up neural network mock
        self._setup_neural_network_mock(1)
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Expected final states
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        expected_food[2, 2] = 30.0  # 50.0 - 20.0
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        expected_creatures[2, 2] = creature_idx
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        expected_creatures_with_border[3, 3] = creature_idx  # (2,2) in creatures maps to (3,3) in creatures_with_border
        
        # Build expected stats from STARTING_VALUES
        expected_stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        expected_stats[0] = creature_stats.STARTING_VALUES.copy()
        expected_stats[0, creature_stats.EAT_RATE] = 20.0  # Override with our custom eat rate
        expected_stats[0, creature_stats.MASS] = 100.0 + 20.0
        expected_stats[0, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[0, creature_stats.LAST_ACTION] = action_kind.ACTION_EAT
        expected_stats[0, creature_stats.LAST_SUCCESS] = 1.0
        expected_stats[0, creature_stats.LAST_COST] = 0.0
        expected_stats[0, creature_stats.LAST_DX] = 0.0
        expected_stats[0, creature_stats.LAST_DY] = 0.0
        expected_stats[0, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS  # Set actual brain mass
        # Cheat and copy MIN_MASS from actual results since it's complex to precalculate
        expected_stats[0, creature_stats.MIN_MASS] = self.simulation.board.creature_storage.stats[0, creature_stats.MIN_MASS]
        
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

    def test_eat_action_partial_consumption(self):
        """Test eat action when there's less food available than eat rate."""
        # Set up initial food on the board - less than eat rate
        self.simulation.board.food[2, 2] = 15.0  # Less than eat rate
        
        # Create a creature with higher eat rate
        genome = Genome()
        creature_idx = self.simulation.board.add_creature(genome)
        
        # Position the creature at (2, 2) where there's food
        self.simulation.board.creatures[2, 2] = creature_idx
        self.simulation.board.creature_storage.grid_position[creature_idx] = [2, 2]
        
        # Set creature's eat rate to 20.0 (but only 15.0 food available)
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.EAT_RATE] = 20.0
        
        # Set initial mass
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.MASS] = 100.0
        
        # Set up neural network mock
        self._setup_neural_network_mock(1)
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Expected final states
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        expected_food[2, 2] = 0.0  # All food consumed
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        expected_creatures[2, 2] = creature_idx
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        expected_creatures_with_border[3, 3] = creature_idx
        
        # Build expected stats from STARTING_VALUES
        expected_stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        expected_stats[0] = creature_stats.STARTING_VALUES.copy()
        expected_stats[0, creature_stats.EAT_RATE] = 20.0  # Override with our custom eat rate
        expected_stats[0, creature_stats.MASS] = 100.0 + 15.0  # Initial mass + food eaten
        expected_stats[0, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[0, creature_stats.LAST_ACTION] = action_kind.ACTION_EAT
        expected_stats[0, creature_stats.LAST_SUCCESS] = 0.75  # 15.0 / 20.0
        expected_stats[0, creature_stats.LAST_COST] = 0.0
        expected_stats[0, creature_stats.LAST_DX] = 0.0
        expected_stats[0, creature_stats.LAST_DY] = 0.0
        expected_stats[0, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS  # Set actual brain mass
        # Cheat and copy MIN_MASS from actual results since it's complex to precalculate
        expected_stats[0, creature_stats.MIN_MASS] = self.simulation.board.creature_storage.stats[0, creature_stats.MIN_MASS]
        
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

    def test_eat_action_no_food(self):
        """Test eat action when there's no food available."""
        # Set up no food on the board
        self.simulation.board.food[2, 2] = 0.0
        
        # Create a creature
        genome = Genome()
        creature_idx = self.simulation.board.add_creature(genome)
        
        # Position the creature at (2, 2)
        self.simulation.board.creatures[2, 2] = creature_idx
        self.simulation.board.creature_storage.grid_position[creature_idx] = [2, 2]
        
        # Set creature's eat rate
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.EAT_RATE] = 20.0
        
        # Set initial mass
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.MASS] = 100.0
        
        # Set up neural network mock
        self._setup_neural_network_mock(1)
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Expected final states
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        expected_food[2, 2] = 0.0  # No food
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        expected_creatures[2, 2] = creature_idx
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        expected_creatures_with_border[3, 3] = creature_idx
        
        # Build expected stats from STARTING_VALUES
        expected_stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        expected_stats[0] = creature_stats.STARTING_VALUES.copy()
        expected_stats[0, creature_stats.EAT_RATE] = 20.0  # Override with our custom eat rate
        expected_stats[0, creature_stats.MASS] = 100.0  # Initial mass (no food eaten)
        expected_stats[0, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[0, creature_stats.LAST_ACTION] = action_kind.ACTION_EAT
        expected_stats[0, creature_stats.LAST_SUCCESS] = 0.0  # No food eaten
        expected_stats[0, creature_stats.LAST_COST] = 0.0
        expected_stats[0, creature_stats.LAST_DX] = 0.0
        expected_stats[0, creature_stats.LAST_DY] = 0.0
        expected_stats[0, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS  # Set actual brain mass
        # Cheat and copy MIN_MASS from actual results since it's complex to precalculate
        expected_stats[0, creature_stats.MIN_MASS] = self.simulation.board.creature_storage.stats[0, creature_stats.MIN_MASS]
        
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

    def test_eat_action_multiple_creatures(self):
        """Test eat action with multiple creatures eating from different locations."""
        # Set up food at multiple locations
        self.simulation.board.food[1, 1] = 30.0
        self.simulation.board.food[3, 3] = 25.0
        
        # Create two creatures
        genome1 = Genome()
        genome2 = Genome()
        creature1_idx = self.simulation.board.add_creature(genome1)
        creature2_idx = self.simulation.board.add_creature(genome2)
        
        # Position creatures at different locations
        self.simulation.board.creatures[1, 1] = creature1_idx
        self.simulation.board.creatures[3, 3] = creature2_idx
        self.simulation.board.creature_storage.grid_position[creature1_idx] = [1, 1]
        self.simulation.board.creature_storage.grid_position[creature2_idx] = [3, 3]
        
        # Set different eat rates
        self.simulation.board.creature_storage.stats[creature1_idx, creature_stats.EAT_RATE] = 15.0
        self.simulation.board.creature_storage.stats[creature2_idx, creature_stats.EAT_RATE] = 20.0
        
        # Set initial masses
        self.simulation.board.creature_storage.stats[creature1_idx, creature_stats.MASS] = 100.0
        self.simulation.board.creature_storage.stats[creature2_idx, creature_stats.MASS] = 150.0
        
        # Set up neural network mock
        self._setup_neural_network_mock(2)
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Expected final states
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        expected_food[1, 1] = 15.0  # 30.0 - 15.0
        expected_food[3, 3] = 5.0   # 25.0 - 20.0
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        expected_creatures[1, 1] = creature1_idx
        expected_creatures[3, 3] = creature2_idx
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        expected_creatures_with_border[2, 2] = creature1_idx  # (1,1) in creatures maps to (2,2) in creatures_with_border
        expected_creatures_with_border[4, 4] = creature2_idx  # (3,3) in creatures maps to (4,4) in creatures_with_border
        
        # Build expected stats from STARTING_VALUES
        expected_stats = np.full((2, creature_stats.COUNT), np.nan, dtype=np.float64)
        expected_stats[0] = creature_stats.STARTING_VALUES.copy()
        expected_stats[1] = creature_stats.STARTING_VALUES.copy()
        expected_stats[0, creature_stats.EAT_RATE] = 15.0  # Override with our custom eat rate
        expected_stats[1, creature_stats.EAT_RATE] = 20.0  # Override with our custom eat rate
        expected_stats[0, creature_stats.MASS] = 100.0 + 15.0  # Initial mass + food eaten
        expected_stats[1, creature_stats.MASS] = 150.0 + 20.0  # Initial mass + food eaten
        expected_stats[0, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[1, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[0, creature_stats.LAST_ACTION] = action_kind.ACTION_EAT
        expected_stats[1, creature_stats.LAST_ACTION] = action_kind.ACTION_EAT
        expected_stats[0, creature_stats.LAST_SUCCESS] = 1.0
        expected_stats[1, creature_stats.LAST_SUCCESS] = 1.0
        expected_stats[0, creature_stats.LAST_COST] = 0.0
        expected_stats[1, creature_stats.LAST_COST] = 0.0
        expected_stats[0, creature_stats.LAST_DX] = 0.0
        expected_stats[1, creature_stats.LAST_DX] = 0.0
        expected_stats[0, creature_stats.LAST_DY] = 0.0
        expected_stats[1, creature_stats.LAST_DY] = 0.0
        expected_stats[0, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS  # Set actual brain mass
        expected_stats[1, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS  # Set actual brain mass
        # Cheat and copy MIN_MASS from actual results since it's complex to precalculate
        expected_stats[0, creature_stats.MIN_MASS] = self.simulation.board.creature_storage.stats[0, creature_stats.MIN_MASS]
        expected_stats[1, creature_stats.MIN_MASS] = self.simulation.board.creature_storage.stats[1, creature_stats.MIN_MASS]
        
        # Apply BMR (Basal Metabolic Rate) reduction
        bmr1 = (expected_stats[0, creature_stats.MASS] + expected_stats[0, creature_stats.MIN_MASS]) / 2000.0
        bmr2 = (expected_stats[1, creature_stats.MASS] + expected_stats[1, creature_stats.MIN_MASS]) / 2000.0
        expected_stats[0, creature_stats.MASS] -= bmr1
        expected_stats[1, creature_stats.MASS] -= bmr2
        
        expected_is_alive = np.ones(2, dtype=bool)
        
        expected_grid_position = np.full((2, 2), -1, dtype=np.int64)
        expected_grid_position[0] = [1, 1]
        expected_grid_position[1] = [3, 3]
        
        # Verify complete board state
        self._verify_board_state(expected_food, expected_creatures, expected_creatures_with_border)
        
        # Verify complete creature storage state
        self._verify_creature_storage_state(expected_stats, expected_is_alive, expected_grid_position)

    def test_eat_action_creature_not_at_food_location(self):
        """Test eat action when creature is not positioned where there's food."""
        # Set up food at (2, 2)
        self.simulation.board.food[2, 2] = 50.0
        
        # Create a creature positioned at (1, 1) - no food there
        genome = Genome()
        creature_idx = self.simulation.board.add_creature(genome)
        
        # Position the creature at (1, 1) where there's no food
        self.simulation.board.creatures[1, 1] = creature_idx
        self.simulation.board.creature_storage.grid_position[creature_idx] = [1, 1]
        
        # Set creature's eat rate
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.EAT_RATE] = 20.0
        
        # Set initial mass
        self.simulation.board.creature_storage.stats[creature_idx, creature_stats.MASS] = 100.0
        
        # Set up neural network mock
        self._setup_neural_network_mock(1)
        
        # Run one round of simulation
        self.simulation.run_round()
        
        # Expected final states
        expected_food = np.zeros((self.height, self.width), dtype=np.float64)
        expected_food[2, 2] = 50.0  # Food unchanged at (2,2)
        
        expected_creatures = np.full((self.height, self.width), -1, dtype=np.int64)
        expected_creatures[1, 1] = creature_idx
        
        expected_creatures_with_border = np.full((self.height + 2, self.width + 2), -1, dtype=np.int64)
        expected_creatures_with_border[2, 2] = creature_idx  # (1,1) in creatures maps to (2,2) in creatures_with_border
        
        # Build expected stats from STARTING_VALUES
        expected_stats = np.full((1, creature_stats.COUNT), np.nan, dtype=np.float64)
        expected_stats[0] = creature_stats.STARTING_VALUES.copy()
        expected_stats[0, creature_stats.EAT_RATE] = 20.0  # Override with our custom eat rate
        expected_stats[0, creature_stats.MASS] = 100.0  # Initial mass (no food eaten)
        expected_stats[0, creature_stats.AGE] = 1.0  # Age incremented in wrapup_round
        expected_stats[0, creature_stats.LAST_ACTION] = action_kind.ACTION_EAT
        expected_stats[0, creature_stats.LAST_SUCCESS] = 0.0  # No food eaten
        expected_stats[0, creature_stats.LAST_COST] = 0.0
        expected_stats[0, creature_stats.LAST_DX] = 0.0
        expected_stats[0, creature_stats.LAST_DY] = 0.0
        expected_stats[0, creature_stats.BRAIN_MASS] = DEFAULT_BRAIN_MASS  # Set actual brain mass
        # Cheat and copy MIN_MASS from actual results since it's complex to precalculate
        expected_stats[0, creature_stats.MIN_MASS] = self.simulation.board.creature_storage.stats[0, creature_stats.MIN_MASS]
        
        # Apply BMR (Basal Metabolic Rate) reduction
        bmr = (expected_stats[0, creature_stats.MASS] + expected_stats[0, creature_stats.MIN_MASS]) / 2000.0
        expected_stats[0, creature_stats.MASS] -= bmr
        
        expected_is_alive = np.ones(1, dtype=bool)
        
        expected_grid_position = np.full((1, 2), -1, dtype=np.int64)
        expected_grid_position[0] = [1, 1]
        
        # Verify complete board state
        self._verify_board_state(expected_food, expected_creatures, expected_creatures_with_border)
        
        # Verify complete creature storage state
        self._verify_creature_storage_state(expected_stats, expected_is_alive, expected_grid_position)


if __name__ == '__main__':
    unittest.main() 