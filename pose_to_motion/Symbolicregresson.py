import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor
import pandas as pd
from loader import load_data
from tqdm import tqdm
import json
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cosine

# Define paths
MODEL_SAVE_DIR = 'models/symbolic_models'
EQUATIONS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'equations2.json')
PLOT_PATH = os.path.join(MODEL_SAVE_DIR, 'symbolic_regression_results2.png')
METRICS_PATH = os.path.join(MODEL_SAVE_DIR, 'symbolic_regression_metrics2.json')

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def plot_results(equations, metrics, save_path=None):
    plt.figure(figsize=(15, 10))
    
    # Plot R² scores
    plt.subplot(2, 2, 1)
    plt.bar(range(len(metrics['r2_scores'])), metrics['r2_scores'])
    plt.xlabel('Output Dimension')
    plt.ylabel('R² Score')
    plt.title('R² Scores for Each Output Dimension')
    plt.grid(True)
    
    # Plot MAE scores
    plt.subplot(2, 2, 2)
    plt.bar(range(len(metrics['mae_scores'])), metrics['mae_scores'])
    plt.xlabel('Output Dimension')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE for Each Output Dimension')
    plt.grid(True)
    
    # Plot Cosine Similarities
    plt.subplot(2, 2, 3)
    plt.bar(range(len(metrics['cosine_similarities'])), metrics['cosine_similarities'])
    plt.xlabel('Output Dimension')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity for Each Output Dimension')
    plt.grid(True)
    
    # Plot Equation Complexities
    complexities = [len(str(eq)) for eq in equations]
    plt.subplot(2, 2, 4)
    plt.bar(range(len(complexities)), complexities)
    plt.xlabel('Output Dimension')
    plt.ylabel('Equation Complexity (Length)')
    plt.title('Complexity of Discovered Equations')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    plt.show()

def symbolic_regression(pose_dataset, motion_dataset):
    ensure_dir(EQUATIONS_SAVE_PATH)
    ensure_dir(PLOT_PATH)
    
    print(f"Input shape: {pose_dataset.shape}, Output shape: {motion_dataset.shape}")
    assert pose_dataset.shape[0] == motion_dataset.shape[0], "Data dimensions mismatch"
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        pose_dataset, motion_dataset, test_size=0.2, random_state=42
    )
    
    # Initialize containers for results
    equations = []
    r2_scores = []
    mae_scores = []
    cosine_similarities = []
    

    
    # Run symbolic regression for each output dimension
    for i in tqdm(range(motion_dataset.shape[1]), desc="Training models"):
        print(f"\n{'='*50}")
        print(f"Training model for output dimension {i+1}/{motion_dataset.shape[1]}")
        
        # Create and configure the symbolic regression model
        model = PySRRegressor(
            niterations=40,  # Number of iterations
            batching =True,  # Enable batching
            batch_size=1000,  # Batch size for training
            binary_operators=["+", "-", "*", "/"],  
            unary_operators=["exp", "sin", "cos", "log", "sqrt", "abs", "tanh", "sinh", "cosh"],  # Unary functions
            populations=10,  # Number of populations
            population_size=25,  # Size of each population
            maxsize=15,  # Maximum complexity of equations
            loss="loss(x, y) = (x - y)^2",  # MSE loss
            procs=0,  # Use all available cores
            parsimony=0.001,  # Regularization for complexity
            warm_start=False,  # Start fresh for each output
            verbosity=1,
            temp_equation_file=f"temp_eq_{i}.csv"
        )
        
        # Fit the model
        model.fit(X_train, y_train[:, i])
        
        # Get the best equation
        best_equation = str(model.sympy())
        equations.append(best_equation)
        print(f"Best equation for output {i}: {best_equation}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test[:, i], y_pred)
        mae = mean_absolute_error(y_test[:, i], y_pred)
        
        # Calculate cosine similarity
        if np.all(y_test[:, i] == 0) or np.all(y_pred == 0):
            cos_sim = 0  # Handle zero vectors
        else:
            cos_sim = 1 - cosine(y_test[:, i], y_pred)
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        cosine_similarities.append(cos_sim)
        
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Cosine Similarity: {cos_sim:.4f}")
    
    # Save results
    results = {
        "equations": equations,
        "r2_scores": r2_scores,
        "mae_scores": mae_scores,
        "cosine_similarities": cosine_similarities,
        "avg_r2": np.mean(r2_scores),
        "avg_mae": np.mean(mae_scores),
        "avg_cosine_similarity": np.mean(cosine_similarities)
    }
    
    with open(EQUATIONS_SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nOverall Performance:")
    print(f"Average R² Score: {results['avg_r2']:.4f}")
    print(f"Average MAE: {results['avg_mae']:.4f}")
    print(f"Average Cosine Similarity: {results['avg_cosine_similarity']:.4f}")
    
    # Plot and save results
    plot_results(equations, results, save_path=PLOT_PATH)
    
    return equations, results

if __name__ == "__main__":
    print("Loading data...")
    pose_dataset, motion_dataset = load_data()
    print(f"Data loaded with shapes: {pose_dataset.shape}, {motion_dataset.shape}")
    
    equations, metrics = symbolic_regression(pose_dataset, motion_dataset)
