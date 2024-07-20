import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchmetrics
import logging
import copy
from focal_loss.focal_loss import FocalLoss
from datetime import datetime
import sys

import parse
from randomGuess import random_guess_baseline
from HGNNP import HGNP
from HGNN import HGN
from utils import seed_setting, weighted_cross_entropy, weights, gpu_config, setup_logging
from baseline.GCN import GCN
from baseline.GAT import GAT
import test
args = parse.parse_arguments()
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"

setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

seed_setting()
model = 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10)


def hgn_training(model, data):
    model.train()
    optimizer.zero_grad()

    # Get the hypergraph output
    out = model(data.node_features, data.hg)
    out_logits = out[data.train_mask]
    logits_shifted = out_logits - \
        out_logits.max(dim=1, keepdim=True).values

    # Get target labels for the training set
    target = data.y[data.train_mask].squeeze().long()
    if target.dim() == 2 and target.shape[1] == 1:
        target = target.squeeze(1)
    probabilities = F.softmax(logits_shifted, dim=1)
    probabilities = torch.clamp(probabilities, min=0, max=1)
    if torch.any(torch.isnan(probabilities)):
        raise RuntimeError(
            "NaN values detected in probabilities after clamping.")

    focal = FocalLoss(gamma=2)
    loss_weighted = weighted_cross_entropy(out_logits, target, weights)

    loss_focal = focal(probabilities, target)
    loss = loss_weighted + loss_focal

    if torch.isnan(loss):
        logging.info("NaN detected in loss computation!")
        return torch.tensor(float('nan'))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss


def graph_train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    target = data.y[data.train_mask].squeeze()
    target = target.long()
    if torch.isnan(out).any():
        logging.info("NaN detected in model output!")
        return torch.tensor(float('nan'))

    loss = F.cross_entropy(out[data.train_mask], target)
    if torch.isnan(loss):
        logging.info("NaN detected in loss computation!")
        logging.info("Model output (first few values):", out[:5])
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
        val_loss = weighted_cross_entropy(
            out[data.val_mask], target, weights)
    return val_loss.item()


    # Train and logging.info the best model
for epoch in range(1, 501):
    train_loss = train(model, data, optimizer)
    val_loss = validate(model, data)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())

    logging.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")
    if epoch % 10 == 0:
        logging.info(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

# Load the best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Test the best model
    (
        test_acc,
        test_f1,
        micro_f1,
    ) = test(model, data)
    logging.info(
        f"Test Metrics - Accuracy: {test_acc:.4f}, F1 Score (Macro): {test_f1:.4f}")
    logging.info(f"Micro F1 Score: {micro_f1:.4f}")


def final_train():
    model, data = HGN()
    y_true = data.y[data.test_mask].cpu().numpy()
    rand_acc, rand_prec, rand_rec, rand_f1 = random_guess_baseline(y_true)
    hgn_training(model, data)


if __name__ == "__main__":
    final_train()
