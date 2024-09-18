import torch
from tqdm import tqdm
from randomGuess import random_guess_baseline
from sklearn.metrics import f1_score, accuracy_score
import torchmetrics
import parse_arg
import logging
from datetime import datetime
from utils import setup_logging
from models.model_utils import get_models
import sys

args = parse_arg.parse_arguments()
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"{args.save_dir}/{time}"

setup_logging(output_folder, console="debug")
logging.debug(" ".join(sys.argv))
logging.debug(f"Arguments: {args}")
logging.debug(f"The testing outputs are being saved in {output_folder}")

def test(model=None, data=None, model_path=None):
    # Load best model if not provided
    if model_path:
        if model is None or data is None:
            model, data = get_models(
                args.model
            )  # Initialize model and data if not provided
        model.load_state_dict(torch.load(model_path))
    else:
        logging.error("Model path is not provided.")
        return

    # Test the model
    model.eval()

    with torch.no_grad():
        out = model(data.node_features, data.hg)
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

    auroc = torchmetrics.AUROC(num_classes=16, task="multiclass")
    auc_score = auroc(pred[data.test_mask], target)
    auc_score = auroc.compute()

    # Micro metrics
    micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")

    # Macro metrics
    test_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
    test_acc = accuracy_score(y_true_flat, y_pred_flat)
    rand_acc, rand_f1_micro, rand_auc, rand_f1 = random_guess_baseline(y_true)
    logging.debug(
        f"Random Guess Metrics - Accuracy: {rand_acc:.4f}, Precision: {rand_f1_micro:.4f}, Recall: {rand_auc:.4f}, F1 Score: {rand_f1:.4f}"
    )

    logging.debug(
        f"Test Accuracy: {test_acc:.4f}, Test F1 Score (Macro): {test_f1:.4f}, Test Micro F1 Score: {micro_f1:.4f}, Test AUC Score: {auc_score:.4f}"
    )

    return test_acc, test_f1, micro_f1, auc_score


if __name__ == "__main__":
    args = parse_arg.parse_arguments()
    test(model_path=args.test_model_path)
