from typing import Tuple
import numpy as np
from neural_network import NeuralNetwork
import copy
import creature_stats
import warnings
import network_outputs
from execution_timer import timer_decorator

BIGGEST_MAGNITUDE_ALLOWED = np.float64(2.0**26.0)
SMALLEST_MAGNITUDE_ALLOWED = 1/BIGGEST_MAGNITUDE_ALLOWED
BIGGEST_EXPONENT_ALLOWED = np.float64(700.0)
SMALLEST_EXPONENT_ALLOWED = 1/BIGGEST_EXPONENT_ALLOWED

class MutationControl:
    """A mutable named tuple of factors that control genome mutation."""

    INTERVAL_LOSE_VISION = 0
    INVERVAL_GAIN_VISION = 1
    INTERVAL_LOSE_NEURON = 2
    INTERVAL_GAIN_NEURON = 3
    INTERVAL_GAIN_LAYER = 4
    INTERVAL_COUNT = 5

    INITIAL_POINT_MUTSUP = 1.0

    @timer_decorator('MutationControl.__init__')
    def __init__(self, like_self_features, like_perception_features, like_params, like_network):
        self.self_feature_coeff_mutsup = np.full_like(like_self_features, self.INITIAL_POINT_MUTSUP)
        self.self_feature_bias_mutsup = np.full_like(like_self_features, self.INITIAL_POINT_MUTSUP)
        self.perception_feature_coeff_mutsup = np.full_like(like_perception_features, self.INITIAL_POINT_MUTSUP)
        self.perception_feature_bias_mutsup = np.full_like(like_perception_features, self.INITIAL_POINT_MUTSUP)
        self.param_coeff_mutsup = np.full_like(like_params, self.INITIAL_POINT_MUTSUP)
        self.network_mutsup = NeuralNetwork(like_network.layer_sizes, constant_value=self.INITIAL_POINT_MUTSUP)

        self.interval_mutation_rates = np.full(self.INTERVAL_COUNT, 0.5, dtype=np.float64)

        self.meta_mutsup = self.INITIAL_POINT_MUTSUP

        self.average_mutsup = self.total_mutsup() / self.mutsup_point_count()
    
    def total_mutsup(self) -> np.float64:
        total = np.sum(self.self_feature_coeff_mutsup) \
            + np.sum(self.self_feature_bias_mutsup) \
            + np.sum(self.perception_feature_coeff_mutsup) \
            + np.sum(self.perception_feature_bias_mutsup) \
            + np.sum(self.param_coeff_mutsup)
        for weights, biases in zip(self.network_mutsup.weights, self.network_mutsup.biases):
            total += np.sum(weights) + np.sum(biases)
        return total
    
    def mutsup_point_count(self) -> int:
        total = self.self_feature_coeff_mutsup.size + self.self_feature_bias_mutsup.size \
            + self.perception_feature_coeff_mutsup.size + self.perception_feature_bias_mutsup.size \
            + self.param_coeff_mutsup.size
        for weights, biases in zip(self.network_mutsup.weights, self.network_mutsup.biases):
            total += weights.size + biases.size
        return total

class Genome:
    """Heritable information to create a creature."""

    _PROPORTIONAL_NOISE_SCALE = 1.0
    _LINEAR_NOISE_SCALE = 0.1
    _EXPONENTIAL_NOISE_SCALE = 0.005

    @timer_decorator('Genome.__init__')
    def __init__(self):
        self.vision_radius = 3 # TODO: temporary universal fixed vision radius

        self.self_feature_coefficients = np.concatenate([
            np.ones(creature_stats.PRIVATE_LINEAR_FEATURES_END - creature_stats.PRIVATE_LINEAR_FEATURES_START),
            np.full(creature_stats.PRIVATE_LOG_FEATURES_END - creature_stats.PRIVATE_LOG_FEATURES_START, 0.2),
            np.ones(creature_stats.PRIVATE_MASS_FRACTION_FEATURES_END - creature_stats.PRIVATE_MASS_FRACTION_FEATURES_START),
            np.ones(creature_stats.PRIVATE_MAX_HP_FRACTION_FEATURES_END - creature_stats.PRIVATE_MAX_HP_FRACTION_FEATURES_START),
            np.ones(creature_stats.PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END - creature_stats.PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START),
        ])

        self.self_feature_biases = np.zeros_like(self.self_feature_coefficients)

        self.perception_feature_coefficients = np.array([[[
            0.2, # food
            0.2, # mass
            1.0, # damage / max hp
        ]]])

        self.perception_feature_biases = np.zeros_like(self.perception_feature_coefficients)

        self.param_coefficients = np.concatenate([
            np.ones(network_outputs.PARAMS_SNAP_DIR_END - network_outputs.PARAMS_SNAP_DIR_START),
            np.ones(network_outputs.PARAMS_TANH_END - network_outputs.PARAMS_TANH_START),
            np.ones(network_outputs.PARAMS_MASS_FRACTIONS_END - network_outputs.PARAMS_MASS_FRACTIONS_START),
        ])

        self.network = NeuralNetwork([self.num_features(), 1, network_outputs.COUNT])

        self.mutation_control = MutationControl(
            self.self_feature_coefficients,
            self.perception_feature_coefficients,
            self.param_coefficients,
            self.network,
        )

    @timer_decorator('Genome.feature_coefficients')
    def feature_coefficients(self) -> np.ndarray:
        return np.concatenate([
            self.self_feature_coefficients,
            self.perception_feature_coefficients.flatten(),
        ])
    
    @timer_decorator('Genome.feature_biases')
    def feature_biases(self) -> np.ndarray:
        return np.concatenate([
            self.self_feature_biases,
            self.perception_feature_biases.flatten(),
        ])

    def num_features(self) -> int:
        return self.self_feature_coefficients.size + self.perception_feature_coefficients.size
    
    _BRAIN_MASS_PER_FLOAT = 0.0001
    _BRAIN_MASS_PER_VISION_RADIUS_CUBED = 0.0 # TODO: penalize larger vision radii once it's evolvable
    
    @timer_decorator('Genome.brain_mass')
    def brain_mass(self) -> np.float64:
        float_count = self.feature_coefficients().size + self.feature_biases().size
        for weights, biases in zip(self.network.weights, self.network.biases):
            float_count += weights.size + biases.size
        network_contribution = self._BRAIN_MASS_PER_FLOAT * float_count
        vision_contribution = self.vision_radius ** 3 * self._BRAIN_MASS_PER_VISION_RADIUS_CUBED
        return network_contribution + vision_contribution
    
    def mutate(self) -> None:
        self._mutate_network_architecture()
        # self._mutate_vision() # TODO: temporary universal fixed vision radius

        self._mutate_positive(self.self_feature_coefficients, self.mutation_control.self_feature_coeff_mutsup, SMALLEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_positive(self.perception_feature_coefficients, self.mutation_control.perception_feature_coeff_mutsup, SMALLEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_positive(self.param_coefficients, self.mutation_control.param_coeff_mutsup, SMALLEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_real(self.self_feature_biases, self.mutation_control.self_feature_bias_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_real(self.perception_feature_biases, self.mutation_control.perception_feature_bias_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        for weights, weights_mutsup, biases, biases_mutsup in zip(self.network.weights, self.mutation_control.network_mutsup.weights, self.network.biases, self.mutation_control.network_mutsup.biases):
            self._mutate_real(weights, weights_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
            self._mutate_real(biases, biases_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)

        self._mutate_positive(self.mutation_control.self_feature_coeff_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        self._mutate_positive(self.mutation_control.self_feature_bias_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        self._mutate_positive(self.mutation_control.perception_feature_coeff_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        self._mutate_positive(self.mutation_control.perception_feature_bias_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        for weights, weights_mutsup, biases, biases_mutsup in zip(self.network.weights, self.mutation_control.network_mutsup.weights, self.network.biases, self.mutation_control.network_mutsup.biases):
            self._mutate_positive(weights_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
            self._mutate_positive(biases_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        
        self._mutate_interval(self.mutation_control.interval_mutation_rates, self.mutation_control.meta_mutsup)

        self.mutation_control.meta_mutsup = self._mutate_positive(np.array([self.mutation_control.meta_mutsup]), self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)[0]

    def _mutate_network_architecture(self) -> None:
        """Mutate the neural network architecture by potentially adding/removing layers and neurons."""
        # Possibly insert an identity layer
        if np.random.random() < self.mutation_control.interval_mutation_rates[self.INTERVAL_GAIN_LAYER]:
            layer_idx = np.random.randint(1, len(self.network.layer_sizes))
            self.network.insert_identity_layer(layer_idx)
            # Mirror the layer insertion in the suppression network with constant suppression
            self.mutation_control.network_mutsup.insert_identity_layer(layer_idx, constant_value=self.mutation_control.average_mutsup)

        # For each hidden layer, possibly insert or prune a neuron/layer
        num_layers = len(self.network.layer_sizes)
        layer_idx = 1
        while layer_idx < num_layers - 1:
            # Chance to insert a neuron
            if np.random.random() < self.mutation_control.interval_mutation_rates[self.INTERVAL_GAIN_NEURON]:
                neuron_idx = np.random.randint(self.network.layer_sizes[layer_idx] + 1)
                self.network.insert_neuron(layer_idx, neuron_idx)
                # Mirror the neuron insertion in the suppression network with constant suppression
                self.mutation_control.network_mutsup.insert_neuron(layer_idx, neuron_idx, constant_value=self.mutation_control.average_mutsup)
            # Chance to prune a neuron or the whole layer
            if np.random.random() < self.mutation_control.interval_mutation_rates[self.INTERVAL_LOSE_NEURON]:
                if self.network.layer_sizes[layer_idx] == 1:
                    self.network.prune_layer(layer_idx)
                    # Mirror the layer pruning in the suppression network
                    self.mutation_control.network_mutsup.prune_layer(layer_idx)
                    num_layers -= 1
                    # After pruning, skip incrementing layer_idx to check the new layer at this index
                    continue
                else:
                    neuron_idx = np.random.randint(self.network.layer_sizes[layer_idx])
                    self.network.prune_neuron(layer_idx, neuron_idx)
                    # Mirror the neuron pruning in the suppression network
                    self.mutation_control.network_mutsup.prune_neuron(layer_idx, neuron_idx)
            layer_idx += 1
    
    def _mutate_vision(self) -> None:
        if np.random.random() < self.mutation_control.interval_mutation_rates[self.INTERVAL_GAIN_VISION]:
            self.vision_radius += 1
        if np.random.random() < self.mutation_control.interval_mutation_rates[self.INTERVAL_LOSE_VISION]:
            self.vision_radius -= 1
        if self.vision_radius < 0:
            self.vision_radius = 0

    def _mutate_real(self, values: np.ndarray, mutsup: np.ndarray, min_value: np.ndarray, max_value: np.ndarray) -> None:
        proportional_noise = np.random.normal(0, self._PROPORTIONAL_NOISE_SCALE, values.shape)
        np.multiply(proportional_noise, values, out=proportional_noise)
        linear_noise = np.random.normal(0, self._LINEAR_NOISE_SCALE, values.shape)
        noise = np.add(proportional_noise, linear_noise, out=linear_noise) # invalidates linear_noise
        suppression_factor = np.exp(-mutsup, out=proportional_noise) # invalidates proportional_noise
        np.multiply(noise, suppression_factor, out=noise)
        np.add(values, noise, out=values)
        if np.any(values < min_value) | np.any(values > max_value):
            warnings.warn(f"Real values exceeded reasonable bounds, clipping to [{min_value}, {max_value}]")
            values.clip(min_value, max_value, out=values)

    def _mutate_positive(self, values: np.ndarray, mutsup: np.ndarray, min_value: np.ndarray, max_value: np.ndarray) -> None:
        noise = np.random.normal(0, self._EXPONENTIAL_NOISE_SCALE, values.shape)
        np.divide(noise, mutsup, out=noise)
        np.exp(noise, out=values)
        if np.any(values < min_value) | np.any(values > max_value):
            warnings.warn(f"Positive values exceeded reasonable bounds, clipping to [{min_value}, {max_value}]")
            values.clip(min_value, max_value, out=values)
    
    def _mutate_interval(self, values: np.ndarray, mutsup: np.ndarray) -> None:
        smaller_answer = values < 0.5
        tax_values = indel_suppression_tax_curve(values)
        self._mutate_real(tax_values, mutsup, 0, 1)
        negative_mask = tax_values < 0
        tax_values[negative_mask] = -tax_values[negative_mask]
        smaller_answer[negative_mask] = ~smaller_answer[negative_mask]
        x1, x2 = inverse_indel_suppression_tax_curve(tax_values)
        values[smaller_answer] = x1[smaller_answer]
        values[~smaller_answer] = x2[~smaller_answer]

LARGE_INSERT_SUPPRESSION_TAX_BASE = 1.001
SMALL_INSERT_SUPPRESSION_TAX_BASE = 1.01
SMALL_DELETE_SUPPRESSION_TAX_BASE = 1.01
POINT_MUTATION_SUPPRESSION_TAX_FACTOR = 0.2
_INDEL_CHANCE_FOR_GEO_MEAN_TAX = 0.1

INDEL_TAX_CURVE_CONSTANT = _INDEL_CHANCE_FOR_GEO_MEAN_TAX * (1.0 - _INDEL_CHANCE_FOR_GEO_MEAN_TAX) / \
    ((_INDEL_CHANCE_FOR_GEO_MEAN_TAX - 0.5) ** 2)

def indel_suppression_tax_curve(x: np.ndarray) -> np.ndarray:
    """Calculate the indel suppression tax curve value.
    
    Args:
        x: The indel chance value in (0,1)
        
    Returns:
        np.array_like: The tax curve value in [0,inf)
    """
    return INDEL_TAX_CURVE_CONSTANT * ((x - 0.5) ** 2) / (x * (1.0 - x))

def inverse_indel_suppression_tax_curve(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the inverse of the indel suppression tax curve.
    
    Args:
        y: The tax curve value (must be >= 0)
        
    Returns:
        tuple[np.array_like, np.array_like]: The two indel chance values in (0,1) that produce the given tax value.
        The values are ordered such that the first value is <= 0.5 and the second value is >= 0.5.
        
    Raises:
        ValueError: If y < 0 or if the equation has no real solutions
    """
    if np.any(y < 0):
        raise ValueError(f"No real solutions exist for the given tax value: {y}")
        
    # Solve the quadratic equation derived from the tax curve formula
    # y = INDEL_TAX_CURVE_CONSTANT * ((x - 0.5)^2) / (x * (1-x))
    # Rearranging and solving for x gives us:
    # y * x * (1-x) = INDEL_TAX_CURVE_CONSTANT * (x^2 - x + 0.25)
    # y * x - y * x^2 = INDEL_TAX_CURVE_CONSTANT * x^2 - INDEL_TAX_CURVE_CONSTANT * x + INDEL_TAX_CURVE_CONSTANT * 0.25
    # (y + INDEL_TAX_CURVE_CONSTANT) * x^2 - (y + INDEL_TAX_CURVE_CONSTANT) * x + INDEL_TAX_CURVE_CONSTANT * 0.25 = 0
    
    a = y + INDEL_TAX_CURVE_CONSTANT
    b = -(y + INDEL_TAX_CURVE_CONSTANT)
    c = INDEL_TAX_CURVE_CONSTANT * 0.25
    
    # Use quadratic formula to solve for x
    discriminant = b*b - 4*a*c
    if np.any(discriminant < 0):
        raise ValueError(f"No real solutions exist for the given tax value: {y}")
        
    sqrt_discriminant = np.sqrt(discriminant)
    x1 = (-b - sqrt_discriminant) / (2*a)
    x2 = (-b + sqrt_discriminant) / (2*a)
    
    return (x1, x2)