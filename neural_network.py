import numpy as np
from typing import Optional

class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], constant_value: Optional[float] = None):
        """Initialize a neural network with the given layer sizes.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer.
                        The first number is the input size, the last is the output size.
            constant_value: If provided, initialize all weights and biases to this value.
                          Otherwise, use random initialization for weights and zeros for biases.
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers")
            
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            if constant_value is not None:
                self.weights.append(np.full((layer_sizes[i + 1], layer_sizes[i]), constant_value, dtype=np.float64))
                self.biases.append(np.full(layer_sizes[i + 1], constant_value, dtype=np.float64))
            else:
                # Uniform random initialization between -1 and 1
                self.weights.append(np.random.uniform(-1, 1, (layer_sizes[i + 1], layer_sizes[i]), dtype=np.float64))
                self.biases.append(np.zeros(layer_sizes[i + 1], dtype=np.float64))
            
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            x: Input array of shape (input_size,)
            
        Returns:
            Output array of shape (output_size,)
        """
        if x.shape != (self.layer_sizes[0],):
            raise ValueError(f"Input shape {x.shape} does not match input size {self.layer_sizes[0]}")
            
        # Forward pass through all layers except the last
        for i in range(len(self.weights) - 1):
            x = self.relu(self.weights[i] @ x + self.biases[i])
            
        return self.weights[-1] @ x + self.biases[-1]

    def to_formatted_string(self) -> str:
        """Get a formatted string representation of the network's weights and biases.
        
        Returns:
            str: Multi-line string showing the network structure and parameters
        """
        lines = []
        
        # Show network structure
        lines.append("Network Structure:")
        lines.append(f"Input layer: {self.layer_sizes[0]} neurons")
        for i in range(1, len(self.layer_sizes) - 1):
            lines.append(f"Hidden layer {i}: {self.layer_sizes[i]} neurons")
        lines.append(f"Output layer: {self.layer_sizes[-1]} neurons")
        lines.append("")
        
        # Show weights and biases for each layer
        for i in range(len(self.weights)):
            layer_name = "Hidden" if i < len(self.weights) - 1 else "Output"
            layer_num = i + 1
            lines.append(f"{layer_name} Layer {layer_num}:")
            
            # Format weights matrix
            lines.append(f"Weights {self.weights[i].shape}:")
            for row in self.weights[i]:
                lines.append("  " + " ".join(f"{w:8.3f}" for w in row))
                
            # Format bias vector
            lines.append(f"Biases {self.biases[i].shape}:")
            lines.append("  " + " ".join(f"{b:8.3f}" for b in self.biases[i]))
            lines.append("")
            
        return "\n".join(lines)

    def insert_neuron(self, layer_idx: int, neuron_idx: Optional[int] = None, constant_value: Optional[float] = None) -> None:
        """Insert a new neuron into a hidden layer while preserving network behavior.
        
        Args:
            layer_idx: Index of the hidden layer to insert the neuron into (0-based)
            neuron_idx: Index within the layer where to insert the neuron (0-based).
                       If None, appends to the end of the layer.
            constant_value: If provided, initialize new weights and biases to this value.
                          Otherwise, use zeros.
            
        Raises:
            ValueError: If layer_idx is invalid (must be a hidden layer)
            ValueError: If neuron_idx is out of bounds
        """
        if layer_idx <= 0 or layer_idx >= len(self.layer_sizes) - 1:
            raise ValueError("Can only insert neurons into hidden layers")
            
        # If neuron_idx is None, append to the end
        if neuron_idx is None:
            neuron_idx = self.layer_sizes[layer_idx]
            
        # Validate neuron_idx
        if neuron_idx < 0 or neuron_idx > self.layer_sizes[layer_idx]:
            raise ValueError(f"neuron_idx {neuron_idx} is out of bounds for layer {layer_idx}")
            
        # Update layer sizes
        self.layer_sizes[layer_idx] += 1
        
        # Get the current weights and biases for the layer
        current_weights = self.weights[layer_idx - 1]  # Weights into this layer
        current_biases = self.biases[layer_idx - 1]    # Biases into this layer
        next_weights = self.weights[layer_idx]         # Weights out of this layer
        
        # Create new weights and biases for the new neuron
        if constant_value is not None:
            new_weights_in = np.full((1, current_weights.shape[1]), constant_value)
            new_bias_in = np.full(1, constant_value)
            new_weights_out = np.full(next_weights.shape[0], constant_value)
        else:
            new_weights_in = np.zeros((1, current_weights.shape[1]))
            new_bias_in = np.zeros(1)
            new_weights_out = np.zeros(next_weights.shape[0])
        
        # Insert the new weights and biases at the specified position
        self.weights[layer_idx - 1] = np.insert(current_weights, neuron_idx, new_weights_in, axis=0)
        self.biases[layer_idx - 1] = np.insert(current_biases, neuron_idx, new_bias_in)
        self.weights[layer_idx] = np.insert(next_weights, neuron_idx, new_weights_out, axis=1)

    def prune_neuron(self, layer_idx: int, neuron_idx: int) -> None:
        """Remove a neuron from a hidden layer.
        
        Args:
            layer_idx: Index of the hidden layer to remove the neuron from (0-based)
            neuron_idx: Index within the layer of the neuron to remove (0-based)
        """
        # Update layer sizes
        self.layer_sizes[layer_idx] -= 1
        
        # Remove the neuron's incoming weights and bias
        self.weights[layer_idx - 1] = np.delete(self.weights[layer_idx - 1], neuron_idx, axis=0)
        self.biases[layer_idx - 1] = np.delete(self.biases[layer_idx - 1], neuron_idx)
        
        # Remove the neuron's outgoing weights
        self.weights[layer_idx] = np.delete(self.weights[layer_idx], neuron_idx, axis=1)

    def insert_identity_layer(self, layer_idx: int, constant_value: Optional[float] = None) -> None:
        """Insert an identity hidden layer at the specified position, preserving network output.
        
        Args:
            layer_idx: Index to insert the new layer (between 1 and len(layer_sizes)-1).
                        The new layer will be inserted before the layer at this index.
            constant_value: If provided, initialize new weights and biases to this value.
                          Otherwise, use identity matrix for weights and zeros for biases.
        Raises:
            ValueError: If layer_idx is not a valid position for a hidden layer.
        """
        if layer_idx <= 0 or layer_idx >= len(self.layer_sizes):
            raise ValueError("layer_idx must be between 1 and len(layer_sizes)-1 (not input or after output layer)")
        
        prev_size = self.layer_sizes[layer_idx - 1]
        new_size = prev_size
        
        # Insert new layer size
        self.layer_sizes.insert(layer_idx, new_size)
        
        # Insert weights and biases
        if constant_value is not None:
            # Use constant value for both weights and biases
            W_in = np.full((new_size, prev_size), constant_value, dtype=np.float64)
            b_in = np.full(new_size, constant_value, dtype=np.float64)
        else:
            # Use identity matrix for weights and zeros for biases
            W_in = np.zeros((new_size, prev_size), dtype=np.float64)
            for i in range(new_size):
                W_in[i, i] = 1.0
            b_in = np.zeros(new_size, dtype=np.float64)
            
        # Insert into weights and biases lists
        self.weights.insert(layer_idx - 1, W_in)
        self.biases.insert(layer_idx - 1, b_in)
        # Biases for the next layer remain unchanged

    def prune_layer(self, layer_idx: int) -> None:
        """Remove a hidden layer at the specified index, connecting previous and next layers directly.
        Assumes the caller knows what they're doing (no safety checks).
        Args:
            layer_idx: Index of the hidden layer to remove (must not be input or output layer).
        """
        # Get weights and biases
        W_in = self.weights[layer_idx - 1]      # Weights into the layer to be removed
        b_in = self.biases[layer_idx - 1]       # Biases into the layer to be removed
        W_out = self.weights[layer_idx]         # Weights out of the layer to be removed
        b_out = self.biases[layer_idx]          # Biases out of the layer to be removed

        # Compose the two layers: new weights and biases
        # y = W_out @ relu(W_in @ x + b_in) + b_out
        # For identity or linear layers, relu is linear, so just W_out @ W_in
        new_weights = W_out @ W_in
        new_biases = W_out @ b_in + b_out

        # Replace the two layers with the composed layer
        self.weights[layer_idx - 1] = new_weights
        self.biases[layer_idx - 1] = new_biases

        # Remove the pruned layer's weights and biases
        del self.weights[layer_idx]
        del self.biases[layer_idx]
        del self.layer_sizes[layer_idx]