import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from utils.model_config import seed_setting
import torchmetrics
import torch.nn as nn
# from DHGCN import DHGCN
from model.HGCN import HGCN
from model.HGNNP import HGNP
from model.HGNN import HGN
from model.HGAT import HGAT
import copy
from focal_loss.focal_loss import FocalLoss

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
    
    # Calculate other metrics
    prec = precision_score(
        y_true_flat, random_predictions_flat, average="macro", zero_division=0
    )
    rec = recall_score(
        y_true_flat, random_predictions_flat, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true_flat, random_predictions_flat, average="macro", zero_division=0
    )
    f1_micro = f1_score(
        y_true_flat, random_predictions_flat, average="micro", zero_division=0
    )

    # Correct usage of AUROC for binary classification
    auroc_metric = torchmetrics.AUROC(num_classes=2, pos_label=1, compute_on_step=False)
    auroc_metric.update(random_predictions_tensor.unsqueeze(0), y_true_tensor.long().unsqueeze(0))
    auc_score = auroc_metric.compute()

    print(f"AUC Score: {auc_score}")

    return acc, prec, rec, f1, f1_micro, auc_score

def weighted_cross_entropy(output, target, weights):
    tensor_weights = torch.tensor(weights, dtype=torch.float, device=output.device)
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    return criterion(output, target)

weights = [1.0] * 16
mbti_to_number = {
            "INTJ": 0,
            "ENTJ": 1,
            "INTP": 2,
            "ENTP": 3,
            "INFJ": 4,
            "INFP": 5,
            "ENFJ": 6,
            "ENFP": 7,
            "ISTJ": 8,
            "ESTJ": 9,
            "ISFJ": 10,
            "ESFJ": 11,
            "ISTP": 12,
            "ESTP": 13,
            "ISFP": 14,
            "ESFP": 15,
        }
    
weights[mbti_to_number["INFP"] - 1] = 0.80  
weights[mbti_to_number["INFJ"] - 1] = 0.78
weights[mbti_to_number["INTP"] - 1] = 0.78

# Split dataset to train, test, and train
def main_training_loop(model, data):
    seed_setting()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    # Training loop
    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.node_features, data.hg)
        out_logits = out[data.train_mask]
        logits_shifted = out_logits - out_logits.max(dim=1, keepdim=True).values
        # Get target labels for the training set
        target = data.y[data.train_mask].squeeze().long()
        # target = target.long()
        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        probabilities = F.softmax(logits_shifted, dim=1)
        probabilities = torch.clamp(probabilities, min=0, max=1)
        if torch.any(torch.isnan(probabilities)):
            raise RuntimeError("NaN values detected in probabilities after clamping.")
        # if not torch.all((probabilities >= 0) & (probabilities <= 1)):
        #     print("Probabilities out of range:", probabilities)
        # print("Target shape:", target.shape)
        focal = FocalLoss(gamma=2)
        loss_weighted = weighted_cross_entropy(out_logits, target, weights)
        
        loss_focal = focal(probabilities, target)
        loss = loss_weighted + loss_focal
        
        if torch.isnan(loss):
            print("NaN detected in loss computation!")
            return torch.tensor(float('nan'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss

    # Validation loop
    def validate(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.node_features, data.hg)
            target = data.y[data.val_mask].squeeze()
            target = target.long()
            val_loss = weighted_cross_entropy(out[data.val_mask], target, weights)
        return val_loss.item()

    def test(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.node_features, data.hg)
            # out = model(data)
            pred = torch.softmax(out, dim=1)
            print(f"Shape and dim of pred: {pred.shape}, {pred.dim()}")

        target = data.y[data.test_mask].squeeze()
        target = target.long()
        print(f"Shape and dim of target: {target.shape}, {target.dim()}")

        y_true = target.numpy()
        y_pred_probs = pred[data.test_mask].cpu().numpy()
        y_pred = y_pred_probs.argmax(axis=1)
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        # cm = confusion_matrix(y_true_flat, y_pred_flat)
        # print("Confusion Matrix:\n", cm)

        auroc = torchmetrics.AUROC(num_classes=16, task="multiclass")
        auc_score = auroc(pred[data.test_mask], target)
        auc_score = auroc.compute()
        print(f"AUC Score: {auc_score}")

        # Micro metrics
        micro_precision = precision_score(y_true_flat, y_pred_flat, average="micro")
        micro_recall = recall_score(y_true_flat, y_pred_flat, average="micro")
        micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")

        # Macro metrics
        test_precision = precision_score(
            y_true_flat, y_pred_flat, average="macro", zero_division=0
        )
        test_recall = recall_score(y_true_flat, y_pred_flat, average="macro")
        test_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
        test_acc = accuracy_score(y_true_flat, y_pred_flat)

        return (
            test_acc,
            test_precision,
            test_recall,
            test_f1,
            micro_precision,
            micro_recall,
            micro_f1,
        )

    best_val_loss = float("inf")
    best_model_state = None

    # Train and print the best model
    for epoch in range(1, 501):
        train_loss = train(model, data, optimizer)
        val_loss = validate(model, data)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test the best model
    (
        test_acc,
        test_precision,
        test_recall,
        test_f1,
        micro_precision,
        micro_recall,
        micro_f1,
    ) = test(model, data)
    print(f"Test Metrics - Accuracy: {test_acc:.4f}, F1 Score (Macro): {test_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")

    # Random guess baseline
    # y_true = data.y[data.test_mask].cpu().numpy()
    # rand_acc, _, _, rand_f1, rand_f1_micro, auc = random_guess_baseline(y_true)

    # print(
    #     f"Random Guess - Accuracy: {rand_acc:.4f}, F1 Score: {rand_f1:.4f}, Micro F1 Score: {rand_f1_micro:.4f}"
    # )

    # Save the best model
    # checkpoint = {
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'epoch': epoch,
    #         'loss': best_val_loss,
    #         'accuracy': test_acc,
    #         }
    # file_path = "checkpoints/DHNN.pth"
    # torch.save(checkpoint, file_path)


def final_train():
    # model, data = HGNP()
    model, data = HGN()
    # model, data = DHGNNP()
    # model, data = HGCN()
    # model = load_model(DeepHGNNP,'deep_hgnnp_model.pth', in_channels=384, hid_channels=384, num_classes=17, use_bn=True, drop_rate=0.5)
    # data = load_hyperedges('hypergraph.pkl')
    main_training_loop(model, data)


if __name__ == "__main__":
    final_train()


# def load_hyperedges(filename='hypergraph.pkl'):
#     with open(filename, 'rb') as f:
#         hypergraph = pickle.load(f)
#     return hypergraph

# def load_model(model_class, filename, **kwargs):
#     model = model_class(**kwargs)  # Re-create the model instance
#     model.load_state_dict(torch.load(filename))
#     model.eval()  # Set the model to evaluation mode
#     return model


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=200, verbose=False, delta=0):
#         """
#         :param patience: How long to wait after last time validation loss improved.
#         :param verbose: If True, prints a message for each validation loss improvement.
#         :param delta: Minimum change in the monitored quantity to qualify as an improvement.
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta

#     def __call__(self, val_loss, model):
#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

# def save_checkpoint(self, val_loss, model):
#     """Saves model when validation loss decrease."""
#     if self.verbose:
#         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
#     torch.save(model.state_dict(), 'checkpoint.pt')
#     self.val_loss_min = val_loss
