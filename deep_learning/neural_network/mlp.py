"""
Multilayer Perceptron (MLP) & Backpropagation Engine (Phase 8)
Implements forward propagation, backpropagation, and gradient updates from scratch using NumPy.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class MultilayerPerceptron:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.05):
        """
        layer_sizes: e.g. [input_dim, hidden_dim_1, output_dim]
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Xavier/He Weight Initialization
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes forward pass activations through each layer.
        """
        activations = [X]
        linear_inputs = []
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            linear_inputs.append(z)
            # Output layer uses softmax if multiclass, hidden layers use sigmoid
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
            
        return linear_inputs, activations

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-Entropy Loss"""
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, linear_inputs: List[np.ndarray], activations: List[np.ndarray], y_true: np.ndarray):
        """
        Backpropagation computing gradients and updating weights.
        """
        m = y_true.shape[0]
        # Output gradient for Softmax + Cross Entropy is (y_pred - y_true)
        dZ = activations[-1] - y_true
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.sigmoid_derivative(activations[i])
                
            # Gradient Descent parameter update
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        linear_inputs, activations = self.forward(X)
        loss = self.compute_loss(activations[-1], y)
        self.backward(linear_inputs, activations, y)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, activations = self.forward(X)
        return activations[-1]
