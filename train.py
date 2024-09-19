import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import logging
import copy
from focal_loss.focal_loss import FocalLoss
from datetime import datetime
import sys
from tqdm import tqdm

import parse_arg
from utils import (
    seed_setting,
    weighted_cross_entropy,
    weights,
    setup_logging,
    gpu_config,
)
from test import test
from models.model_utils import get_models

args = parse_arg.parse_arguments()
# Record the training debug
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"{args.save_dir}/{time}"

setup_logging(output_folder, console_level="info", debug_filename="info.log")
# logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The training outputs are being saved in {output_folder}/info.log")


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
        logging.error("NaN values detected in probabilities after clamping.")
        raise RuntimeError

    focal = FocalLoss(gamma=2)
    loss_weighted = weighted_cross_entropy(out_logits, target, weights)

    loss_focal = focal(probabilities, target)
    loss = loss_weighted + loss_focal

    if torch.isnan(loss):
        logging.debug("NaN detected in loss computation!")
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
    seed_setting()

    device=gpu_config()

    best_val_loss = float("inf")
    best_model_state = None

    # Initialize model, optimizer, scheduler
    model, data = get_models(args.model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    # Iterate and output the best model
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        train_loss = train(model, data, optimizer, args.model, device)
        val_loss = validate(model, data, device)
        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # Save the best model
    torch.save(best_model_state, f"{output_folder}/best_model_{time}.pth")
    logging.info(f"Best model saved to {output_folder}/best_model_{time}.pth")
    model.load_state_dict(best_model_state)
    
    # Back to CPU after training
    model = model.to("cpu")
    torch.cuda.empty_cache()

    # Test the best model
    test_acc, test_f1, micro_f1, auc_score = test(model, data)

    return test_acc, test_f1, micro_f1, auc_score


if __name__ == "__main__":
    main()
