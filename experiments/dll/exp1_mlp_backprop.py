"""
DLL Experiment 1: MLP & Backpropagation Visualization
Demonstrates forward pass, Cross-Entropy loss, and gradient descent weight updates.
"""

import numpy as np
from deep_learning.neural_network.mlp import MultilayerPerceptron

def run_mlp_backprop_experiment(epochs: int = 15) -> dict:
    np.random.seed(42)
    # Realistic 2-feature document classification dataset (e.g., Line Count vs Word Density)
    # Class 0: Short receipts / IDs
    X0 = np.random.randn(30, 2) * 0.45 + np.array([-0.9, -0.9])
    # Class 1: Multi-line Invoices / Resumes
    X1 = np.random.randn(30, 2) * 0.45 + np.array([0.9, 0.9])
    
    X = np.vstack([X0, X1])
    y = np.zeros((60, 2))
    y[:30, 0] = 1.0  # Class 0 one-hot
    y[30:, 1] = 1.0  # Class 1 one-hot
    
    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    mlp = MultilayerPerceptron(layer_sizes=[2, 8, 2], learning_rate=0.45)
    loss_history = []
    
    for epoch in range(epochs):
        loss = mlp.train_epoch(X, y)
        loss_history.append({"epoch": epoch + 1, "loss": round(float(loss), 4)})
        
    predictions = mlp.predict(X)
    accuracy = float(np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)))
    
    return {
        "experiment_name": "MLP Feedforward & Backpropagation",
        "architecture": "Input(2) -> Hidden(8, Sigmoid) -> Output(2, Softmax)",
        "epochs": epochs,
        "initial_loss": loss_history[0]["loss"],
        "final_loss": loss_history[-1]["loss"],
        "accuracy": round(accuracy * 100, 2),
        "loss_history": loss_history,
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    result = run_mlp_backprop_experiment()
    print("Initial:", result["initial_loss"], "Final:", result["final_loss"], "Acc:", result["accuracy"])
    for p in result["loss_history"]:
        print(p)
