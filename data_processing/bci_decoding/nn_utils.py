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


def evaluate_model(model, dataloader, loss_fn=nn.MSELoss(), device='cuda'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: PyTorch dataloader
        loss_fn: Loss function
        device: Device to run evaluation on ('cuda' or 'cpu')
        
    Returns:
        float: Average loss on the dataset
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return avg_loss


def train_model(model, dataloader, optimizer, scheduler, num_iterations, 
               loss_fn=nn.MSELoss(), device='cuda', val_dataloader=None, val_interval=100,
               noise_neural_std=None, noise_bias_std=None):
    """
    Training loop for a PyTorch model with optional validation evaluation.
    
    Args:
        model: PyTorch model
        dataloader: PyTorch dataloader for training
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        num_iterations: Number of training iterations
        loss_fn: Loss function
        device: Device to run training on ('cuda' or 'cpu')
        val_dataloader: PyTorch dataloader for validation (optional, set to None to skip validation)
        val_interval: Number of iterations between validation evaluations
        noise_neural_std: std of white noise to add to neural data
        noise_bias_std: std of bias noise to add to neural data
        
    Returns:
        model: Trained model
        dict: Training history with losses and validation losses (if validation was used)
    """
    model.to(device)
    model.train()
    
    # Initialize history dictionary
    history = {
        'train_losses': [],
        'iterations': []
    }
    
    # Add validation losses list only if validation is enabled
    if val_dataloader is not None:
        history['val_losses'] = []
    
    iteration = 0
    pbar = tqdm(total=num_iterations, desc="Training")
    
    while iteration < num_iterations:
        for inputs, targets in dataloader:
            if iteration >= num_iterations:
                break
                
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            if noise_neural_std or noise_bias_std:
                inputs = add_training_noise(inputs, noise_neural_std, noise_bias_std, device=device)
                
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            model.train()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Record loss
            history['train_losses'].append(loss.item())
            history['iterations'].append(iteration)
            
            # Update learning rate
            scheduler.step()
            
            # Validation
            if val_dataloader is not None and iteration % val_interval == 0:
                val_loss = evaluate_model(model, val_dataloader, loss_fn, device)
                history['val_losses'].append(val_loss)
                
                # Log validation performance
                pbar.set_postfix({
                    "train_loss": f"{loss.item():.4f}",
                    "val_loss": f"{val_loss:.4f}"
                })
            else:
                pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            iteration += 1
            pbar.update(1)
            
    pbar.close()
    
    return model, history

def add_training_noise(x,
                       bias_neural_std=None,
                       noise_neural_std=None,
                       noise_neural_walk_std=None,
                       bias_allchans_neural_std=None,
                       device='cpu'):
    """Function to add different types of noise to training input data to make models more robust.
       Identical to the methods in Willet 2021.
    Args:
        x (tensor):                     neural data of shape [batch_size x seq_len x num_chans]
        bias_neural_std (float):        std of bias noise
        noise_neural_std (float):       std of white noise
        noise_neural_walk_std (float):  std of random walk noise
        bias_allchans_neural_std (float): std of bias noise, bias is same across all channels
        device (device):                torch device (cpu or cuda)
    """
    # Transpose x to [batch_size x num_chans x seq_len] to match original function's expectation
    x = x.transpose(1, 2)
    
    if bias_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), but different for each channel & batch
        biases = torch.normal(torch.zeros(x.shape[:2]), bias_neural_std).unsqueeze(2).repeat(1, 1, x.shape[2])
        x = x + biases.to(device=device)

    if noise_neural_std:
        # adds white noise to each channel and timepoint (independent)
        noise = torch.normal(torch.zeros_like(x), noise_neural_std)
        x = x + noise.to(device=device)

    if noise_neural_walk_std:
        # adds a random walk to each channel (noise is summed across time)
        noise = torch.normal(torch.zeros_like(x), noise_neural_walk_std).cumsum(dim=2)
        x = x + noise.to(device=device)

    if bias_allchans_neural_std:
        # bias is constant across time (i.e. the 3 conv inputs), and same for each channel
        biases = torch.normal(torch.zeros((x.shape[0], 1, 1)), bias_allchans_neural_std).repeat(1, x.shape[1], x.shape[2])
        x = x + biases.to(device=device)

    # Transpose back to [batch_size x seq_len x num_chans]
    x = x.transpose(1, 2)
    
    return x


