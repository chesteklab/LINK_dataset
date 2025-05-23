import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def calculate_mse_per_dof(predictions, targets):
    """
    Calculate Mean Squared Error for each DOF.
    
    Args:
        predictions: numpy array of shape (n_samples, n_dofs)
        targets: numpy array of shape (n_samples, n_dofs)
        
    Returns:
        numpy array: MSE values for each DOF
    """
    n_dofs = predictions.shape[1]
    mse_values = np.zeros(n_dofs)
    
    for i in range(n_dofs):
        mse_values[i] = np.mean((predictions[:, i] - targets[:, i]) ** 2)
    
    return mse_values

def calculate_correlation_per_dof(predictions, targets):
    """
    Calculate Pearson correlation for each DOF.
    
    Args:
        predictions: numpy array of shape (n_samples, n_dofs)
        targets: numpy array of shape (n_samples, n_dofs)
        
    Returns:
        numpy array: correlation values for each DOF
    """
    n_dofs = predictions.shape[1]
    correlations = np.zeros(n_dofs)
    
    for i in range(n_dofs):
        # Calculate Pearson correlation and only keep the correlation value (first element of tuple)
        correlations[i], _ = pearsonr(predictions[:, i], targets[:, i])
    
    return correlations

def calculate_r2_per_dof(predictions, targets):
    """
    Calculate R2 score for each DOF.
    
    Args:
        predictions: numpy array of shape (n_samples, n_dofs)
        targets: numpy array of shape (n_samples, n_dofs)
        
    Returns:
        numpy array: R2 scores for each DOF
    """
    n_dofs = predictions.shape[1]
    r2_scores = np.zeros(n_dofs)
    
    for i in range(n_dofs):
        r2_scores[i] = r2_score(targets[:, i], predictions[:, i])
    
    return r2_scores 