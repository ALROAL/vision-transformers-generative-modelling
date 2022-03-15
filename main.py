import argparse
import os
import sys

from src.models.train_model import main as train


def main(
    name: str = "test",
    model_type: str = "Classifier",
    max_epochs: int = 10,
    num_workers: int = 0,
    dim: int = 1024,
    depth: int = 12,
    heads: int = 16,
    mlp_dim: int = 2048,
    lr: float = 3e-5
):

    train(
        name=name,
        model_type=model_type,
        max_epochs=max_epochs,
        num_workers=num_workers,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        lr=lr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a model (ViT-VAE or Classifier) on CIFAR10"
    )
    parser.add_argument("--name", "-n", type=str, help="Name for wandb")
    parser.add_argument(
        "--model-type",
        "-mt",
        type=str,
        help="Model type to train either ViTVAE or Classifier",
    )
    parser.add_argument("--max-epochs", "-me", type=int, help="Number of max epochs")
    parser.add_argument(
        "--num-workers", "-nw", type=int, help="Number of threads use in loading data"
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="Last dimension of output tensor after linear transformation",
    )
    parser.add_argument("--depth", type=int, help="Number of Transformer blocks")
    parser.add_argument(
        "--heads", type=int, help="Number of heads in Multi-head Attention layer"
    )
    parser.add_argument(
        "--mlp_dim", type=int, help="Dimension of the MLP (FeedForward) layer"
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate (Currently only for classifier)"
    )
    args = parser.parse_args()

    name = "test"
    model_type = "Classifier"
    max_epochs = 10
    num_workers = 0
    dim = 1024
    depth = 12
    heads = 16
    mlp_dim = 2048
    lr = 3e-5

    if args.name:
        name = args.name
    if args.model_type:
        model_type = args.model_type
    if args.max_epochs:
        max_epochs = args.max_epochs
    if args.num_workers:
        num_workers = args.num_workers
    if args.dim:
        dim = args.dim
    if args.depth:
        depth = args.depth
    if args.heads:
        heads = args.heads
    if args.mlp_dim:
        mlp_dim = args.mlp_dim
    if args.lr:
        lr = args.lr

    main(
        name=name,
        model_type=model_type,
        max_epochs=max_epochs,
        num_workers=num_workers,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        lr=lr
    )
