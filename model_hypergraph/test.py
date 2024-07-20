import torch
from tqdm import tqdm
from model_hypergraph.utils import seed_setting
from train import random_guess_baseline
from sklearn.metrics import f1_score, accuracy_score
import torchmetrics
import argparse
import logging

def test(model, data):
    model.eval()
    
    with torch.no_grad():
        out = model(data.node_features, data.hg)
        pred = torch.softmax(out, dim=1)
        logging.info(f"Shape and dim of pred: {pred.shape}, {pred.dim()}")

    target = data.y[data.test_mask].squeeze()
    target = target.long()
    logging.info(f"Shape and dim of target: {target.shape}, {target.dim()}")

    y_true = target.numpy()
    y_pred_probs = pred[data.test_mask].cpu().numpy()
    y_pred = y_pred_probs.argmax(axis=1)
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    auroc = torchmetrics.AUROC(num_classes=16, task="multiclass")
    auc_score = auroc(pred[data.test_mask], target)
    auc_score = auroc.compute()
    logging.info(f"AUC Score: {auc_score}")

    # Micro metrics
    micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")

    # Macro metrics
    test_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
    test_acc = accuracy_score(y_true_flat, y_pred_flat)

    return (
        test_acc,
        test_f1,
        micro_f1,
    )
