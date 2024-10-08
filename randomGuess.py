import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
import torchmetrics
from utils import seed_setting

# Random guess
def random_guess_baseline(y_true):
    seed_setting()

    # Generate random predictions with the same shape as y_true
    random_predictions = np.random.randint(0, 2, size=y_true.shape)

    # Flatten the arrays for multilabel classification
    y_true_flat = y_true.ravel()
    random_predictions_flat = random_predictions.ravel()

    # Convert numpy arrays to PyTorch tensors
    y_true_tensor = torch.tensor(y_true_flat, dtype=torch.float)
    random_predictions_tensor = torch.tensor(random_predictions_flat, dtype=torch.float)

    # Calculate accuracy for flattened arrays
    acc = accuracy_score(y_true_flat, random_predictions_flat)

    f1 = f1_score(
        y_true_flat, random_predictions_flat, average="macro", zero_division=0
    )
    f1_micro = f1_score(
        y_true_flat, random_predictions_flat, average="micro", zero_division=0
    )

    # Correct usage of AUROC for classification
    auroc_metric = torchmetrics.AUROC(
        num_classes=2,
        task="binary",
    )
    auroc_metric.update(
        random_predictions_tensor.unsqueeze(0),
        y_true_tensor.long().unsqueeze(0),
    )
    auc_score = auroc_metric.compute()

    return acc, f1, f1_micro, auc_score
