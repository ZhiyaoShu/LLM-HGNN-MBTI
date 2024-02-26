
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from utils import seed_setting, save_model
import torchmetrics
from DHGCN import DHGCN
import torch.optim as optim
import copy
import tensorflow as tf

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


# Random guess
def random_guess_baseline(y_true):
    seed_setting()

    # Generate random predictions with the same shape as y_true
    random_predictions = np.random.randint(0, 2, size=y_true.shape)

    # Flatten the arrays for multilabel classification
    y_true_flat = y_true.ravel()
    random_predictions_flat = random_predictions.ravel()

    # Calculate accuracy for flattened arrays
    acc = accuracy_score(y_true_flat, random_predictions_flat)

    # Calculate other metrics
    prec = precision_score(y_true_flat, random_predictions_flat, average='macro', zero_division=0)
    rec = recall_score(y_true_flat, random_predictions_flat, average='macro', zero_division=0)
    f1 = f1_score(y_true_flat, random_predictions_flat, average='macro', zero_division=0)

    return acc, prec, rec, f1

# Split dataset to train, test, and train
def main_training_loop(model, data):
    seed_setting()

    # print(f"criterion: {criterion}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)  
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.node_features, data.hg)

        # print("Output shape:", out.shape)
               
        # train_outputs = out[data.train_mask]
        # print("Train outputs shape:", train_outputs.shape)
        # Get target labels for the training set
        target = data.y[data.train_mask].squeeze()
        target = target.long()

        # print("Target shape:", target.shape)
                
        loss = F.cross_entropy(out[data.train_mask], target)
        # print(out[data.train_mask].shape) 
        # print(f"Loss grad_fn: {loss.grad_fn}")
        # print(f"Model output grad_fn: {out.grad_fn}")  
        # print(target.dtype)
        # print(out.isfinite().all())
        # print(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss
    
    def validate(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.node_features, data.hg)
            target = data.y[data.val_mask].squeeze()
            target = target.long()
            val_loss = F.cross_entropy(out[data.val_mask], target)
        return val_loss.item()

    def test(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.node_features, data.hg)
            pred = torch.softmax(out, dim=1)
            
        target = data.y[data.test_mask].squeeze()
        target = target.long() 
        y_true = target.cpu().numpy()
        y_pred_probs = pred[data.test_mask].cpu().numpy()
        y_pred = y_pred_probs.argmax(axis=1)
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        # cm = confusion_matrix(y_true_flat, y_pred_flat)
        # print("Confusion Matrix:\n", cm)
        
        auroc = torchmetrics.AUROC(num_classes=17, task="multiclass")
        auc_score = auroc(pred[data.test_mask], target)
        auc_score = auroc.compute()
        print(f"AUC Score: {auc_score}")
        
        # Micro metrics
        micro_precision = precision_score(y_true_flat, y_pred_flat, average='micro')
        micro_recall = recall_score(y_true_flat, y_pred_flat, average='micro')
        micro_f1 = f1_score(y_true_flat, y_pred_flat, average='micro')
        
        # Macro metrics
        test_precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        test_recall = recall_score(y_true_flat, y_pred_flat, average='macro')
        test_f1 = f1_score(y_true_flat, y_pred_flat, average='macro')
        test_acc = accuracy_score(y_true_flat, y_pred_flat)
        
        return test_acc, test_precision, test_recall, test_f1, micro_precision, micro_recall, micro_f1
    
    best_val_loss = float('inf')
    # early_stopping = EarlyStopping(patience=10, verbose=True)
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
        if epoch%10 == 0:        
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # early_stopping(val_loss, model)
    if best_model_state:
        model.load_state_dict(best_model_state)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
    test_acc, test_precision, test_recall, test_f1, micro_precision, micro_recall, micro_f1 = test(model, data)
    print(f'Test Metrics - Accuracy: {test_acc:.4f}, F1 Score (Macro): {test_f1:.4f}')
    print(f'Micro F1 Score: {micro_f1:.4f}')

    y_true = data.y[data.test_mask].cpu().numpy()
    rand_acc, rand_prec, rand_rec, rand_f1 = random_guess_baseline(y_true)
    print(f'Random Guess - Accuracy: {rand_acc:.4f}') 
    
    checkpoint2 = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': best_val_loss,
            'accuracy': test_acc,
            }
    
    torch.save(checkpoint2, 'model_checkpoint.pth2') 
    

def final_train():
    model, data = DHGCN()
    main_training_loop(model, data)
    # torch.save(model.state_dict(), 'HGCN.pth')
    
    
if __name__ == "__main__":  
    final_train()
    
