import numpy as np
from neural_network import NeuralNetwork
import copy
import creature_stats
import warnings
import network_outputs

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

    def __init__(self, like_private_features, like_public_features, like_params, like_network):
        self.private_feature_coeff_mutsup = np.full_like(like_private_features, self.INITIAL_POINT_MUTSUP)
        self.private_feature_bias_mutsup = np.full_like(like_private_features, self.INITIAL_POINT_MUTSUP)
        self.public_feature_coeff_mutsup = np.full_like(like_public_features, self.INITIAL_POINT_MUTSUP)
        self.public_feature_bias_mutsup = np.full_like(like_public_features, self.INITIAL_POINT_MUTSUP)
        self.param_coeff_mutsup = np.full_like(like_params, self.INITIAL_POINT_MUTSUP)
        self.network_mutsup = NeuralNetwork(like_network.layer_sizes, constant_value=self.INITIAL_POINT_MUTSUP)

        self.interval_mutation_rates = np.full(self.INTERVAL_COUNT, 0.5, dtype=np.float64)

        self.meta_mutsup = self.INITIAL_POINT_MUTSUP
    
    def total_mutsup(self) -> np.float64:
        total = np.sum(self.private_feature_coeff_mutsup) \
            + np.sum(self.private_feature_bias_mutsup) \
            + np.sum(self.public_feature_coeff_mutsup) \
            + np.sum(self.public_feature_bias_mutsup) \
            + np.sum(self.param_coeff_mutsup)
        for weights, biases in zip(self.network_mutsup.weights, self.network_mutsup.biases):
            total += np.sum(weights) + np.sum(biases)
        return total

class Genome:
    """Heritable information to create a creature."""

    PROPORTIONAL_NOISE_SCALE = 1.0
    LINEAR_NOISE_SCALE = 0.1
    EXPONENTIAL_NOISE_SCALE = 0.005

    def __init(self):
        self.vision_radius = 0

        self.private_feature_coefficients = np.concatenate([
            np.ones(creature_stats.PRIVATE_LINEAR_FEATURES_END - creature_stats.PRIVATE_LINEAR_FEATURES_START),
            np.full(creature_stats.PRIVATE_LOG_FEATURES_END - creature_stats.PRIVATE_LOG_FEATURES_START, 0.1),
            np.ones(creature_stats.PRIVATE_MASS_FRACTION_FEATURES_END - creature_stats.PRIVATE_MASS_FRACTION_FEATURES_START),
            np.ones(creature_stats.PRIVATE_MAX_HP_FRACTION_FEATURES_END - creature_stats.PRIVATE_MAX_HP_FRACTION_FEATURES_START),
            np.ones(creature_stats.PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_END - creature_stats.PRIVATE_LAST_DAMAGE_FRACTION_FEATURES_START),
        ])

        self.private_feature_biases = np.zeros_like(self.private_feature_coefficients)

        self.public_feature_coefficients = np.array([[[
            0.1, # food
            0.1, # mass
            1.0, # damage / max hp
        ]]])

        self.public_feature_biases = np.zeros_like(self.public_feature_coefficients)

        self.param_coefficients = np.concatenate([
            np.ones(network_outputs.PARAMS_SNAP_DIR_END - network_outputs.PARAMS_SNAP_DIR_START),
            np.ones(network_outputs.PARAMS_TANH_END - network_outputs.PARAMS_TANH_START),
            np.ones(network_outputs.PARAMS_MASS_FRACTIONS_END - network_outputs.PARAMS_MASS_FRACTIONS_START),
        ])

        self.network = NeuralNetwork([self.num_features(), 1, network_outputs.COUNT])

        self.mutation_control = MutationControl(
            self.private_feature_coefficients,
            self.public_feature_coefficients,
            self.param_coefficients,
            self.network,
        )

    def num_features(self) -> int:
        return self.private_feature_coefficients.size + self.public_feature_coefficients.size
    
    def mutate(self) -> None:
        self._mutate_network_architecture()

        self._mutate_positive(self.private_feature_coefficients, self.mutation_control.private_feature_coeff_mutsup, SMALLEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_positive(self.public_feature_coefficients, self.mutation_control.public_feature_coeff_mutsup, SMALLEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_positive(self.param_coefficients, self.mutation_control.param_coeff_mutsup, SMALLEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_real(self.private_feature_biases, self.mutation_control.private_feature_bias_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        self._mutate_real(self.public_feature_biases, self.mutation_control.public_feature_bias_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
        for weights, weights_mutsup, biases, biases_mutsup in zip(self.network.weights, self.mutation_control.network_mutsup.weights, self.network.biases, self.mutation_control.network_mutsup.biases):
            self._mutate_real(weights, weights_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)
            self._mutate_real(biases, biases_mutsup, -BIGGEST_MAGNITUDE_ALLOWED, BIGGEST_MAGNITUDE_ALLOWED)

        self._mutate_positive(self.mutation_control.private_feature_coeff_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        self._mutate_positive(self.mutation_control.private_feature_bias_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        self._mutate_positive(self.mutation_control.public_feature_coeff_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        self._mutate_positive(self.mutation_control.public_feature_bias_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        for weights, weights_mutsup, biases, biases_mutsup in zip(self.network.weights, self.mutation_control.network_mutsup.weights, self.network.biases, self.mutation_control.network_mutsup.biases):
            self._mutate_positive(weights_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
            self._mutate_positive(biases_mutsup, self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)
        
        self._mutate_interval(self.mutation_control.interval_mutation_rates, self.mutation_control.meta_mutsup)

        self.mutation_control.meta_mutsup = self._mutate_positive(np.array([self.mutation_control.meta_mutsup]), self.mutation_control.meta_mutsup, SMALLEST_EXPONENT_ALLOWED, BIGGEST_EXPONENT_ALLOWED)[0]

    def _mutate_network_architecture(self) -> None:
        pass

    def _mutate_real(self, values: np.ndarray, mutsup: np.ndarray, min_value: np.array_like, max_value: np.array_like) -> None:
        proportional_noise = np.random.normal(0, self.PROPORTIONAL_NOISE_SCALE, values.shape)
        np.multiply(proportional_noise, values, out=proportional_noise)
        linear_noise = np.random.normal(0, self.LINEAR_NOISE_SCALE, values.shape)
        noise = np.add(proportional_noise, linear_noise, out=linear_noise) # invalidates linear_noise
        suppression_factor = np.exp(-mutsup, out=proportional_noise) # invalidates proportional_noise
        np.multiply(noise, suppression_factor, out=noise)
        np.add(values, noise, out=values)
        if np.any(values < min_value) | np.any(values > max_value):
            warnings.warn(f"Real values exceeded reasonable bounds, clipping to [{min_value}, {max_value}]")
            values.clip(min_value, max_value, out=values)

    def _mutate_positive(self, values: np.ndarray, mutsup: np.array_like, min_value: np.array_like, max_value: np.array_like) -> None:
        noise = np.random.normal(0, self.EXPONENTIAL_NOISE_SCALE, values.shape)
        np.divide(noise, mutsup, out=noise)
        np.exp(noise, out=values)
        if np.any(values < min_value) | np.any(values > max_value):
            warnings.warn(f"Positive values exceeded reasonable bounds, clipping to [{min_value}, {max_value}]")
            values.clip(min_value, max_value, out=values)
    
    def _mutate_interval(self, values: np.ndarray, mutsup: np.array_like) -> None:
        smaller_answer = values < 0.5
        tax_values = indel_suppression_tax_curve(values)
        self._mutate_real(tax_values, mutsup, 0, 1)
        negative_mask = tax_values < 0
        tax_values[negative_mask] = -tax_values[negative_mask]
        smaller_answer[negative_mask] = ~smaller_answer[negative_mask]
        x1, x2 = inverse_indel_suppression_tax_curve(tax_values)
        values[smaller_answer] = x1[smaller_answer]
        values[~smaller_answer] = x2[~smaller_answer]

INDEL_CHANCE_FOR_GEO_MEAN_TAX = 0.1
LARGE_INSERT_SUPPRESSION_TAX_BASE = 1.001
SMALL_INSERT_SUPPRESSION_TAX_BASE = 1.01
SMALL_DELETE_SUPPRESSION_TAX_BASE = 1.01
POINT_MUTATION_SUPPRESSION_TAX_FACTOR = 0.2

INDEL_TAX_CURVE_CONSTANT = INDEL_CHANCE_FOR_GEO_MEAN_TAX * (1.0 - INDEL_CHANCE_FOR_GEO_MEAN_TAX) / \
    ((INDEL_CHANCE_FOR_GEO_MEAN_TAX - 0.5) ** 2)

def indel_suppression_tax_curve(x: np.array_like) -> np.array_like:
    pass

def inverse_indel_suppression_tax_curve(y: np.array_like) -> tuple[np.array_like, np.array_like]:
    pass