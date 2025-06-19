import os
import torch
import optuna
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from improvedModel import ImprovedMotionModel
from loader import load_data

MODEL_SAVE_PATH = 'models/models/improve_motion_model_test.pth'
PLOT_PATH = 'models/pics/loss_plot_test.png'
BEST_PARAMS_FILE = 'models/params/best_params.json'
BEST_METRICS_FILE = 'models/params/best_metrics.json'
OPTUNA_RESULTS_FILE = 'models/params/optuna_results.json'


GLOBAL_BEST_LOSS = float('inf')
GLOBAL_BEST_TRIAL = None

def plot_losses(train_losses, val_losses, val_maes, val_r2_scores, val_cosine_similarities, save_path=None, params=None):

    plt.figure(figsize=(12, 8))

    # Plot Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.plot(val_maes, label='Validation MAE', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Mean Absolute Error (MAE)', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    # Plot R²
    plt.subplot(2, 2, 3)
    plt.plot(val_r2_scores, label='Validation R²', color='purple', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Validation R² Score', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    # Plot Cosine Similarity
    plt.subplot(2, 2, 4)
    plt.plot(val_cosine_similarities, label='Validation Cosine Similarity', color='brown', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Validation Cosine Similarity', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    plt.tight_layout()

    if params:
        param_text = "\n".join([f"{key}: {value}" for key, value in params.items()])
        plt.gcf().text(0.95, 0.95, param_text, fontsize=10, color='black',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=0.8))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Loss plot saved to {save_path}")
    plt.close()

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping, in case of exploding gradients
        optimizer.step() # Update model parameters
        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(loader.dataset)

    return avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_mae = 0.0
    total_cosine_similarity = 0.0
    total_ss_res = 0.0
    total_ss_tot = 0.0
    cosine_sim = nn.CosineSimilarity(dim=1)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            total_mae += torch.mean(torch.abs(outputs - targets)).item() * inputs.size(0)
            total_cosine_similarity += cosine_sim(outputs, targets).sum().item()

            total_ss_res += torch.sum((targets - outputs) ** 2).item()
            total_ss_tot += torch.sum((targets - targets.mean(dim=0)) ** 2).item()
    avg_loss = running_loss / len(loader.dataset)
    avg_mae = total_mae / len(loader.dataset)
    avg_cosine_similarity = total_cosine_similarity / len(loader.dataset)
    r2_score = 1 - total_ss_res / total_ss_tot

    return avg_loss, avg_mae, avg_cosine_similarity, r2_score

def save_best_results(model, train_losses, val_losses, val_maes, val_cosine_similarities, 
                     val_r2_scores, trial_params, trial_number, final_epoch, best_val_loss):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    best_params = {
        "trial_number": trial_number,
        "best_val_loss": best_val_loss,
        "final_epoch": final_epoch,
        **trial_params
    }
    
    os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)

    best_metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes,
        "val_cosine_similarities": val_cosine_similarities,
        "val_r2_scores": val_r2_scores,
        "best_val_loss": best_val_loss,
        "final_epoch": final_epoch,
        "trial_number": trial_number
    }
    
    os.makedirs(os.path.dirname(BEST_METRICS_FILE), exist_ok=True)
    with open(BEST_METRICS_FILE, 'w') as f:
        json.dump(best_metrics, f, indent=4)

    params_for_plot = {
        "Best Trial": trial_number,
        "Best Val Loss": f"{best_val_loss:.6f}",
        "Learning Rate": f"{trial_params['learning_rate']:.2e}",
        "Batch Size": trial_params['batch_size'],
        "Weight Decay": f"{trial_params['weight_decay']:.2e}",
        "Final Epoch": final_epoch,
        "Dropout": trial_params['dropout']
    }
    
    plot_losses(train_losses, val_losses, val_maes, val_r2_scores, 
               val_cosine_similarities, save_path=PLOT_PATH, params=params_for_plot)
    
    
def objective (trial, pose_dataset, motion_dataset):
    global GLOBAL_BEST_LOSS, GLOBAL_BEST_TRIAL 
    # Define the hyperparameters to optimize
    batch_size = trial.suggest_int("batch_size", 64, 256, step=32)

    num_epochs = trial.suggest_int("num_epochs", 150, 300)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)
    hidden_sizes = [
        trial.suggest_int(f"hidden_size_{i}", 16, 128, step=16) for i in range(4)
    ]  
    trial_number = trial.number
    model_save_path=MODEL_SAVE_PATH
    plot_path=PLOT_PATH
    assert pose_dataset.shape[0] == motion_dataset.shape[0], "Sample count mismatch"
    
    pose_train, pose_val, motion_train, motion_val = train_test_split(
        pose_dataset, motion_dataset, test_size=0.2, random_state=42
    )
    # Convert to tensors
    pose_train_tensor = torch.tensor(pose_train, dtype=torch.float32)
    motion_train_tensor = torch.tensor(motion_train, dtype=torch.float32)
    pose_val_tensor = torch.tensor(pose_val, dtype=torch.float32)
    motion_val_tensor = torch.tensor(motion_val, dtype=torch.float32)

    # Datasets and loaders
    train_dataset = TensorDataset(pose_train_tensor, motion_train_tensor)
    val_dataset = TensorDataset(pose_val_tensor, motion_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = train_dataset[0][0].shape[0]
    output_size = train_dataset[0][1].shape[0]

    print(f"Input size: {input_size}, Output size: {output_size}")
    
    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedMotionModel(input_size, output_size, hidden_sizes, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) #when the validation loss plateaus, reduce the learning rate by 50%

    # early stopping parameters in case of overfitting
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    # Training loop
    train_losses, val_losses = [], []
    val_maes = []
    val_cosine_similarities = []
    val_r2_scores = []
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_cosine_similarity, val_r2 = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_cosine_similarities.append(val_cosine_similarity)
        val_r2_scores.append(val_r2)

        scheduler.step(val_loss)
        trial.report(val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), model_save_path)
            print(f"Model improved and saved at epoch {epoch}, val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:03}/{num_epochs}, "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Val MAE: {val_mae:.6f}, "
              f"Val Cosine Similarity: {val_cosine_similarity:.6f}, "
              f"Val R2 Score: {val_r2:.6f}")
        
    if best_val_loss < GLOBAL_BEST_LOSS:
        GLOBAL_BEST_LOSS = best_val_loss
        GLOBAL_BEST_TRIAL = trial_number

        trial_params = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout
        }
        
        save_best_results(
            model, train_losses, val_losses, val_maes, 
            val_cosine_similarities, val_r2_scores, 
            trial_params, trial_number, epoch, best_val_loss
        )

    else:
        print(f"   Trial {trial_number} finished: {best_val_loss:.6f} "
              f"({GLOBAL_BEST_LOSS:.6f} from Trial {GLOBAL_BEST_TRIAL})")

    return best_val_loss

if __name__ == "__main__":
    print("Loading data once at the beginning...")
    pose_dataset, motion_dataset = load_data()
    print(f"Data loaded with shapes: {pose_dataset.shape}, {motion_dataset.shape}")

    def objective_with_data(trial):
        return objective(trial, pose_dataset, motion_dataset)
     
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_with_data, n_trials=5)

    best_params = study.best_params
    best_value = study.best_value
    print("Best hyperparameters:", best_params)
    print("Best validation loss:", best_value)

    optuna_results = {
        "best_params": best_params,
        "best_value": best_value,
        "best_trial_number": GLOBAL_BEST_TRIAL,
        "total_trials": len(study.trials),
        "all_trial_values": [trial.value for trial in study.trials if trial.value is not None]
    }
    
    os.makedirs(os.path.dirname(OPTUNA_RESULTS_FILE), exist_ok=True)
    with open(OPTUNA_RESULTS_FILE, 'w') as f:
        json.dump(optuna_results, f, indent=4)





   
