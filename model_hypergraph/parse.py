import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parse_arguments(
        "--model",
        type=str,
        default="hgnn",
        help="The model to use. Can be hgnn, hgnnp, GCN, GAT, GTransformer. Default is HGNN.",
    )

    parser.add_argument(
        "--llm",
        type=bool,
        default=True,
        help="Whether to use LLM pre-processed embeddings. Default is True.",
    )
    
    parser.add_argument(
        "--num_classes",
        type=int,
        default=16,
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
        "--lr", type=float,
        default=0.1,
        help="The learning rate. Default is 0.1."
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="The number of training epochs. Default is 500.",
    )
