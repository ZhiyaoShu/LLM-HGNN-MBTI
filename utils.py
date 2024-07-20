import random
import numpy as np
import torch
import torch.nn as nn
import logging
import os
import multiprocessing


# Random seeds
def seed_setting(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Weight loss functions
def weighted_cross_entropy(output, target, weights):
    tensor_weights = torch.tensor(
        weights, dtype=torch.float, device=output.device)
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


# Set up ouput directory
def setup_logging(output_folder, console="debug"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_file = os.path.join(output_folder, "output.log")

    logging.basicConfig(
        level=logging.DEBUG if console == "debug" else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# Check if GPU is available
def gpu_config():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_numbers = torch.cuda.device_count()
        logging.info(f"Using {gpu_numbers} GPUs")
    else:
        device = torch.device("cpu")
        cpu_numbers = multiprocessing.cpu_count()
        logging.info(f"Using {cpu_numbers} CPU")
    return device
