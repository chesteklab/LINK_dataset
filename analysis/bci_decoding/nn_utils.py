import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SequenceScaler:
    """
    Wrapper around StandardScaler to handle neural data of shape (n_samples, seq_len, n_features)
    Ensures same normalization is applied across all timesteps.
    Fits just on the last timestep of the sequence.
    
    X is of shape (n_samples, seq_len, n_features)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X):
        self.scaler.fit(X[:, -1, :])
        return self
    
    def transform(self, X):
        # loop over each timestep and apply the scaler
        for i in range(X.shape[1]):
            X[:, i, :] = self.scaler.transform(X[:, i, :])
        return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class FingerDataset(Dataset):
    """
    Dataset for finger kinematics and neural data.
    """
    def __init__(self, neural_data, kinematics_data):
        # neural_data is a numpy array of shape (n_samples, seq_len, n_channels)
        # kinematics_data is a numpy array of shape (n_samples, 2 * n_fingers)
        self.neural_data = torch.tensor(neural_data, dtype=torch.float32)
        self.kinematics_data = torch.tensor(kinematics_data, dtype=torch.float32)

    def __len__(self):
        return len(self.neural_data)

    def __getitem__(self, idx): 
        return self.neural_data[idx], self.kinematics_data[idx]


def create_optimizer_and_scheduler(model, lr=0.001, weight_decay=0, final_lr=None, 
                                  total_steps=None):
    """
    Creates an Adam optimizer and linear learning rate scheduler for a model.
    
    Args:
        model: PyTorch model
        lr: Initial learning rate
        weight_decay: L2 regularization factor
        final_lr: Final learning rate after decay (if None, defaults to 0.1*lr)
        total_steps: Total number of steps for the scheduler (if None, defaults to 1000)
        
    Returns:
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set defaults if not provided
    if final_lr is None:
        final_lr = 0.1 * lr
    if total_steps is None:
        total_steps = 1000
    
    # Calculate end_factor from final_lr
    end_factor = final_lr / lr
    
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=end_factor, 
        total_iters=total_steps
    )
    
    return optimizer, scheduler


def train_model(model, dataloader, optimizer, scheduler, num_iterations, 
               loss_fn=nn.MSELoss(), device='cuda'):
    """
    Simple training loop for a PyTorch model.
    
    Args:
        model: PyTorch model
        dataloader: PyTorch dataloader
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        num_iterations: Number of training iterations
        loss_fn: Loss function
        device: Device to run training on ('cuda' or 'cpu')
        
    Returns:
        model: Trained model
        losses: List of training losses
    """
    model.to(device)
    model.train()
    losses = []
    
    iteration = 0
    pbar = tqdm(total=num_iterations, desc="Training")
    while iteration < num_iterations:
        for inputs, targets in dataloader:
            if iteration >= num_iterations:
                break
                
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Update learning rate
            scheduler.step()
            
            iteration += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    pbar.close()
    return model, losses




