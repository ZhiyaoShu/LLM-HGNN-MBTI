import torch
from tqdm import tqdm
from randomGuess import random_guess_baseline
from sklearn.metrics import f1_score, accuracy_score
import torchmetrics
import parse_arg
import logging
from datetime import datetime

# from utils import setup_logging
from models.model_utils import get_models
import sys

args = parse_arg.parse_arguments()
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"{args.save_dir}/{time}"


def test(model, model_type, data, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.test_model_path
    
    # Load best model if not provided
    if model_path and isinstance(model_path, str):
        if model is None or data is None:
            model, data = get_models(args.model)
        model.load_state_dict(torch.load(model_path))
    elif isinstance(model_path, dict):
        model.load_state_dict(model_path)

    model = model.to(device)

    data.node_features = data.node_features.to(device)
    if model_type in ["hgnn", "hgnnp"]:
        data.hg = data.hg.to(device)

    # Test the model
    model.eval()

    with torch.no_grad():
        if model_type in ["hgnn", "hgnnp"]:
            out = model(data.node_features, data.hg)
        else:
            out = model(data)
        pred = torch.softmax(out, dim=1)
        logging.info(f"Shape and dim of pred: {pred.shape}, {pred.dim()}")

    target = data.y[data.test_mask].squeeze().long().to(device)
    target = target.long()
    logging.info(f"Shape and dim of target: {target.shape}, {target.dim()}")

    y_true = target.cpu().numpy()
    y_pred_probs = pred[data.test_mask].cpu().numpy()

    y_pred = []
    for prob in tqdm(y_pred_probs, desc="Inferencing", leave=True):
        y_pred.append(prob.argmax())
    y_true_flat = y_true.ravel()
    y_pred_flat = torch.tensor(y_pred).numpy().ravel()

    auroc = torchmetrics.AUROC(num_classes=16, task="multiclass").to(device)
    auc_score = auroc(pred[data.test_mask], target)
    auc_score = auroc.compute()

    # Micro metrics
    micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")

    # Macro metrics
    test_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
    test_acc = accuracy_score(y_true_flat, y_pred_flat)
    rand_acc, rand_f1_micro, rand_auc, rand_f1 = random_guess_baseline(y_true)
    logging.info(
        f"""Random Guess Metrics 
                Accuracy: {rand_acc:.4f}, 
                Precision: {rand_f1_micro:.4f}, 
                Recall: {rand_auc:.4f}, 
                F1 Score: {rand_f1:.4f}
            """
    )

    logging.info(
        f"""Test Accuracy: {test_acc:.4f}, 
                Test F1 Score (Macro): {test_f1:.4f}, 
                Test Micro F1 Score: {micro_f1:.4f}, 
                Test AUC Score: {auc_score:.4f}"""
    )

    model = model.to("cpu")
    torch.cuda.empty_cache()

    return test_acc, test_f1, micro_f1, auc_score
