import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hgnn",
        help="The model to use. Can be hgnn, hgnnp, gcn, gat, gtrans. Default is hgnn.",
    )

    parser.add_argument(
        "--use_llm",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Whether to use LLM pre-processed embeddings. Default is True.",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=16, # 16 MBTI types
        help="The number of HGCN layers. Default is 16.",
    )

    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=300,
        help="The number of hidden channels. Default is 300.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="The dropout probability. Default is 0.5.",
    )

    parser.add_argument(
        "--lr", type=float, default=0.1, help="The learning rate. Default is 0.1."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="The number of training epochs. Default is 500.",
    )

    parser.add_argument(
        "--mbti",
        type=bool,
        default=True,
        help="Use MBTI labels or Enneagram(False). Default is True.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs",
        help="Directory to save logs and model. Default is current directory.",
    )

    parser.add_argument(
        "--test_model_path",
        type=str,
        default=None,
        help="Path to the model to test directly. Default is None.",
    )

    args = parser.parse_args()
    return args
