import numpy as np
import torch

def add_history(neural_data, seq_len):
    """
    Add history to the neural data.
    neural_data is of shape (n_samples, n_channels)
    the output is of shape (n_samples, seq_len, n_channels)
    """
    Xtrain1 = torch.zeros((int(neural_data.shape[0]), int(neural_data.shape[1]), seq_len))
    Xtrain1[:, :, 0] = torch.from_numpy(neural_data)
    for k1 in range(seq_len - 1):
        k = k1 + 1
        Xtrain1[k:, :, k] = torch.from_numpy(neural_data[0:-k, :])

    # for RNNs, we want the last timestep to be the most recent data
    Xtrain1 = torch.flip(Xtrain1, (2,))

    # reorder from (n_samples, n_channels, seq_len) to (n_samples, seq_len, n_channels)
    return Xtrain1.permute(0, 2, 1).numpy()


def prep_data_and_split(data_dict, seq_len, num_train_trials, should_add_history=True, verbose=False):
    trial_index = data_dict['trial_index']
    if len(trial_index) > num_train_trials:
        test_len = np.min((len(trial_index)-1, 399))

        neural_training = data_dict['sbp'][:trial_index[num_train_trials]]
        neural_testing = data_dict['sbp'][trial_index[num_train_trials]:trial_index[test_len]]

        if verbose:
            print(f"neural_training.shape: {neural_training.shape}")
            print(f"neural_testing.shape: {neural_testing.shape}")

        finger_training = data_dict['finger_kinematics'][:trial_index[num_train_trials]]
        finger_testing = data_dict['finger_kinematics'][trial_index[num_train_trials]:trial_index[test_len]]

        # add history
        if should_add_history:
            neural_training_hist = add_history(neural_training, seq_len)
            neural_testing_hist = add_history(neural_testing, seq_len)
            if verbose:
                print(f"neural_training_hist.shape: {neural_training_hist.shape}")
                print(f"neural_testing_hist.shape: {neural_testing_hist.shape}")
            return neural_training_hist, neural_testing_hist, finger_training, finger_testing
        else:
            return neural_training, neural_testing, finger_training, finger_testing

    else:
        raise Exception('not enough trials')


def test_add_history():
    """
    Test function to verify the add_history function works correctly
    """
    # Create a small test array with easily recognizable values
    # 5 samples, 2 channels
    test_data = np.array([
        [1, 10],   # Sample 0
        [2, 20],   # Sample 1
        [3, 30],   # Sample 2
        [4, 40],   # Sample 3
        [5, 50]    # Sample 4
    ])
    
    print("Original data (5 samples, 2 channels):")
    print(test_data)
    print()
    
    # Test with sequence length 3
    seq_len = 3
    result = add_history(test_data, seq_len)
    
    print(f"After add_history with seq_len={seq_len} (5 samples, {seq_len} history steps, 2 channels):")
    for i in range(len(result)):
        print(f"Sample {i}:")
        print(result[i])
        print()
    
    # Verify expected outcomes for specific samples
    print("Verification:")
    print(f"Sample 0 should have zeros for history and [1, 10] for current: ")
    print(f"Actual: {result[0]}")
    
    print(f"\nSample 2 should have [1, 10] for oldest, [2, 20] for middle, and [3, 30] for current: ")
    print(f"Actual: {result[2]}")
    
    print(f"\nSample 4 should have [3, 30] for oldest, [4, 40] for middle, and [5, 50] for current: ")
    print(f"Actual: {result[4]}")


if __name__ == "__main__":
    test_add_history()