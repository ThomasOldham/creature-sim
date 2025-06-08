import numpy as np
from genome import Genome

def test_vision_radius_changes():
    # Create initial genome
    genome = Genome()
    initial_radius = genome.vision_radius
    
    # Store initial state
    initial_coeffs = genome.perception_feature_coefficients.copy()
    initial_biases = genome.perception_feature_biases.copy()
    initial_coeff_mutsup = genome.mutation_control.perception_feature_coeff_mutsup.copy()
    initial_bias_mutsup = genome.mutation_control.perception_feature_bias_mutsup.copy()
    initial_network_weights = [w.copy() for w in genome.network.weights]
    initial_network_biases = [b.copy() for b in genome.network.biases]
    initial_network_mutsup_weights = [w.copy() for w in genome.mutation_control.network_mutsup.weights]
    initial_network_mutsup_biases = [b.copy() for b in genome.mutation_control.network_mutsup.biases]
    
    # Increment vision radius
    genome._increment_vision_radius()
    assert genome.vision_radius == initial_radius + 1, "Vision radius should increase by 1"
    
    # Decrement vision radius
    genome._decrement_vision_radius()
    assert genome.vision_radius == initial_radius, "Vision radius should return to initial value"
    
    # Check that all arrays are back to their original state
    assert np.array_equal(genome.perception_feature_coefficients, initial_coeffs), "Feature coefficients should be unchanged"
    assert np.array_equal(genome.perception_feature_biases, initial_biases), "Feature biases should be unchanged"
    assert np.array_equal(genome.mutation_control.perception_feature_coeff_mutsup, initial_coeff_mutsup), "Coefficient mutation suppression should be unchanged"
    assert np.array_equal(genome.mutation_control.perception_feature_bias_mutsup, initial_bias_mutsup), "Bias mutation suppression should be unchanged"
    
    # Check network weights and biases
    for i, (w, b) in enumerate(zip(genome.network.weights, genome.network.biases)):
        assert np.array_equal(w, initial_network_weights[i]), f"Network weights for layer {i} should be unchanged"
        assert np.array_equal(b, initial_network_biases[i]), f"Network biases for layer {i} should be unchanged"
    
    # Check network mutation suppression weights and biases
    for i, (w, b) in enumerate(zip(genome.mutation_control.network_mutsup.weights, genome.mutation_control.network_mutsup.biases)):
        assert np.array_equal(w, initial_network_mutsup_weights[i]), f"Network mutation suppression weights for layer {i} should be unchanged"
        assert np.array_equal(b, initial_network_mutsup_biases[i]), f"Network mutation suppression biases for layer {i} should be unchanged"

if __name__ == "__main__":
    test_vision_radius_changes()
    print("All tests passed!") 