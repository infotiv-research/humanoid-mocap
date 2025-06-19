import os
import torch
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
from simpleModel import SimpleMotionModel
from loader import load_data

MODEL_SAVE_PATH = 'models/models/pose_motion_model.pth'
PLOT_PATH = 'models/pics/loss_plot_posenorm.png'
BEST_PARAMS_FILE = 'models/params/best_params.json'
FINAL_TRAINING_METRICS='models/params/final_training_metrics_posenorm.json'

def ensure_dir(file_path):

    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


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
    plt.show()



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

def train_model(pose_dataset, motion_dataset, params_file = None):

    if params_file and os.path.exists(params_file):
        with open(params_file, 'r') as f:
            results = json.load(f)
        best_params = results["best_params"]
        print(f"Loaded best parameters from {params_file}: {best_params}")
    else:
        best_params = {
            "num_epochs": 150,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "hidden_size_0": 124,
            "hidden_size_1": 64,
            "hidden_size_2": 32,
            "hidden_size_3": 16,
            "batch_size": 90
        }
        print("Using default parameters:", best_params)

    batch_size = best_params["batch_size"]
    num_epochs = best_params["num_epochs"]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    hidden_sizes = []
    i = 0
    while f"hidden_size_{i}" in best_params:
            hidden_sizes.append(best_params[f"hidden_size_{i}"])
            i += 1
    model_save_path=MODEL_SAVE_PATH
    plot_path=PLOT_PATH
    ensure_dir(model_save_path)  
    ensure_dir(plot_path)
    assert pose_dataset.shape[0] == motion_dataset.shape[0]
    print(f"Good match")
    
    pose_train, pose_val, motion_train, motion_val = train_test_split(
        pose_dataset, motion_dataset, test_size=0.2, random_state=42
    )

    #Np array to pytorch tensor
    pose_train_tensor = torch.tensor(pose_train, dtype = torch.float32)
    motion_train_tensor = torch.tensor(motion_train, dtype=torch.float32)
    pose_val_tensor = torch.tensor(pose_val, dtype=torch.float32)
    motion_val_tensor = torch.tensor(motion_val, dtype=torch.float32)
    #Tensor dataset
    train_dataset = TensorDataset(pose_train_tensor, motion_train_tensor)
    val_dataset = TensorDataset(pose_val_tensor, motion_val_tensor)
    #Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    input_size = train_dataset[0][0].shape[0]  
    output_size = train_dataset[0][1].shape[0]
    print("Input size, output size", input_size, output_size)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedMotionModel(input_size, output_size, hidden_sizes, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) #when the validation loss plateaus, reduce the learning rate by 50%

    # early stopping parameters in case of overfitting
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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

    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes,
        "val_cosine_similarities": val_cosine_similarities,
        "val_r2_scores": val_r2_scores,
        "params": best_params
    }
    
    with open(FINAL_TRAINING_METRICS, 'w') as f:
        json.dump(metrics, f, indent=4)

    params_for_plot = {
        "Learning Rate": lr,
        "Batch Size": batch_size,
        "Weight Decay": weight_decay,
        "Epochs": num_epochs,
        "Dropout": dropout
    }
    
    plot_losses(train_losses, val_losses, val_maes, val_r2_scores, val_cosine_similarities, 
                save_path=plot_path, params=params_for_plot)
    
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    final_val_loss, final_mae, final_cosine_similarity, final_r2 = evaluate(model, val_loader, criterion, device)
    print("\nFinal Model Performance:")
    print(f"Validation Loss: {final_val_loss:.6f}")
    print(f"Validation MAE: {final_mae:.6f}")
    print(f"Validation Cosine Similarity: {final_cosine_similarity:.6f}")
    print(f"Validation R2 Score: {final_r2:.6f}")
    return model

if __name__ == "__main__":
    print("Loading data once at the beginning...")
    pose_dataset, motion_dataset = load_data()
    print(f"Data loaded with shapes: {pose_dataset.shape}, {motion_dataset.shape}")
    best_params_file = BEST_PARAMS_FILE
    trained_model = train_model(pose_dataset, motion_dataset, params_file=best_params_file)
