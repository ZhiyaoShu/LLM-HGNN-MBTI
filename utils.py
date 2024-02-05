import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Random seeds
def seed_setting(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_model(model, path, epoch, val_loss):
    """ Saves the model state. """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': val_loss,
    }, path)

def log_metrics(epoch, train_loss, val_loss, test_acc=None):
    """ Logs metrics for monitoring. """
    print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if test_acc:
        print(f"Test Accuracy: {test_acc:.4f}")