
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn.functional as F
from utils import seed_setting
import torchmetrics
from GCN import GCN
from GAT import GAT
from G_transformer import GCNCT
import joblib
import pickle
import copy

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 


    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        target = data.y[data.train_mask].squeeze()
        target = target.long()
                
        loss = F.cross_entropy(out[data.train_mask], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss
    
    def validate(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            target = data.y[data.val_mask].squeeze()
            target = target.long()
            val_loss = F.cross_entropy(out[data.val_mask], target)
        return val_loss.item()

    def test(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = torch.softmax(out, dim=1)
            
        target = data.y[data.test_mask].squeeze()
        target = target.long() 
        y_true = target.cpu().numpy()
        y_pred_probs = pred[data.test_mask].cpu().numpy()
        y_pred = y_pred_probs.argmax(axis=1)
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
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
    print(f'Test Metrics - Accuracy: {test_acc:.4f}, Precision (Macro): {test_precision:.4f}, Recall (Macro): {test_recall:.4f}, F1 Score (Macro): {test_f1:.4f}')
    print(f'Micro Metrics - Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1 Score: {micro_f1:.4f}')

    y_true = data.y[data.test_mask].cpu().numpy()
    rand_acc, rand_prec, rand_rec, rand_f1 = random_guess_baseline(y_true)
    print(f'Random Guess - Accuracy: {rand_acc:.4f}, Precision: {rand_prec:.4f}, Recall: {rand_rec:.4f}, F1 Score: {rand_f1:.4f}')  


def final_train():
    model,data = GAT()
    # model,data = GCN()
    # model,data = GCNCT()
    main_training_loop(model, data)
    
if __name__ == "__main__":  
    final_train()
 
