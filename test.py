import torch
from tqdm import tqdm
from train import random_guess_baseline
from sklearn.metrics import f1_score, accuracy_score
import torchmetrics
import parse
import logging
import datetime

def test(model, data):
    # Load best model
    args = parse.parse_arguments()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"logs/{args.save_dir}/{time}"
    model.load_state_dict(torch.load(f"{output_folder}/best_model_{time}.pth"))
    
    # Test the model
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

    auroc = torchmetrics.AUROC(num_classes=args, task="multiclass")
    auc_score = auroc(pred[data.test_mask], target)
    auc_score = auroc.compute()

    # Micro metrics
    micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")

    # Macro metrics
    test_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
    test_acc = accuracy_score(y_true_flat, y_pred_flat)
    logging.info(
        f"""Test Metrics 
                    Accuracy: {test_acc:.4f},
                    AUC Score: {auc_score}, 
                    F1 Score (Macro): {test_f1:.4f},
                    Micro F1 Score: {micro_f1:.4f}
            """
    )
    rand_acc, rand_prec, rand_rec, rand_f1 = random_guess_baseline(y_true)
    logging.info(
        f"Random Guess Metrics - Accuracy: {rand_acc:.4f}, Precision: {rand_prec:.4f}, Recall: {rand_rec:.4f}, F1 Score: {rand_f1:.4f}"
    )

    return test_acc, test_f1, micro_f1, auc_score
