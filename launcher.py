from ZapfenDataset import ZapfenDataset
from models.ffnn import FFNN
from train import train


def launch():
    train(FFNN())
    # data = ZapfenDataset('./zapfen.csv')


if __name__ == '__main__':
    launch()
