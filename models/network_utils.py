import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import shapiro


# Feature normalization
def check_data_distribution(data):
    """
    Check the distribution of the features in the data.
    """

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
