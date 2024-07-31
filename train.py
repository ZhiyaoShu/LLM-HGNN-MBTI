import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import logging
import copy
from focal_loss.focal_loss import FocalLoss
from datetime import datetime
import sys
import tqdm

import parse_arg
from utils import seed_setting, weighted_cross_entropy, weights, setup_logging
from test import test  # 注意test模块已经被导入
from models.get_models import get_models


def train(model, data, optimizer, model_type):
    model.train()
    optimizer.zero_grad()

    # Get the hypergraph output
    if model_type in ["hgnn", "hgnnp"]:
        out = model(data.node_features, data.hg)
        out_logits = out[data.train_mask]
        logits_shifted = out_logits - out_logits.max(dim=1, keepdim=True).values

    # Get target labels for the training set
    target = data.y[data.train_mask].squeeze().long()
    if target.dim() == 2 and target.shape[1] == 1:
        target = target.squeeze(1)
    probabilities = F.softmax(logits_shifted, dim=1)
    probabilities = torch.clamp(probabilities, min=0, max=1)
    if torch.any(torch.isnan(probabilities)):
        raise RuntimeError("NaN values detected in probabilities after clamping.")

    focal = FocalLoss(gamma=2)
    loss_weighted = weighted_cross_entropy(out_logits, target, weights)

    loss_focal = focal(probabilities, target)
    loss = loss_weighted + loss_focal

    if torch.isnan(loss):
        logging.info("NaN detected in loss computation!")
        return torch.tensor(float("nan"))
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


def main():
    # Record the training info
    args = parse_arg.parse_arguments()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"logs/{args.save_dir}/{time}"

    setup_logging(output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    seed_setting()
    best_val_loss = float("inf")
    best_model_state = None

    # Initialize model, optimizer, scheduler
    model, data, _, _ = get_models(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    # Iterate and output the best model
    for epoch in tqdm.tqdm(range(1, args.epochs + 1), desc="Training Progress"):
        train_loss = train(model, data, optimizer, args.model)
        val_loss = validate(model, data)
        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # Save the best model
    torch.save(best_model_state, f"{output_folder}/best_model_{time}.pth")
    model.load_state_dict(best_model_state)

    # Test the best model
    test_acc, test_f1, micro_f1, auc_score = test(model, data)

    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1 Score (Macro): {test_f1:.4f}")
    logging.info(f"Test Micro F1 Score: {micro_f1:.4f}")
    logging.info(f"Test AUC Score: {auc_score:.4f}")
    logging.info("Experiment Completed")
    return test_acc, test_f1, micro_f1, auc_score


if __name__ == "__main__":
    main()
