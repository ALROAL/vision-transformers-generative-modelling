import os
import sys

from src.models.train_model import main as train


def main():
    name = sys.argv[1]
    model_type = sys.argv[2]
    max_epochs = int(sys.argv[3])

    train(name=name, model_type=model_type, max_epochs=max_epochs)


if __name__ == "__main__":
    main()
