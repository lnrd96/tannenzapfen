import torch

NUM_EPOCHS = 10

DATALOADER_CONFIG = {'batch_size': 8}

NUM_FEATURES = 16
NUM_CLASSES = 4

TRAIN_CONFIG = {
    'loss_fn': torch.nn.CrossEntropyLoss(),  # mutual exclusive classes
}

NN_CONFIG = {'act_func': torch.nn.ReLU(),
             'out_func': torch.nn.Softmax(dim=1)}
