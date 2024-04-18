import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Random seeds
def seed_setting(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def check_data_distribution(data):
    """
    Check the distribution of the features in the data.
    """
    from scipy.stats import shapiro

    sample = data.x[np.random.choice(data.x.size(0), 1000, replace=False)].numpy()

    stat, p = shapiro(sample)
    return "normal" if p > 0.05 else "non-normal"

def normalize_features(data):
    """
    Normalize the node features based on their distribution.
    """
    distribution = check_data_distribution(data)
    scaler = StandardScaler() if distribution == "normal" else MinMaxScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float)
    return data


def log_metrics(epoch, train_loss, val_loss, test_acc=None):
    """ Logs metrics for monitoring. """
    print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if test_acc:
        print(f"Test Accuracy: {test_acc:.4f}")