import torch

NUM_EPOCHS = 100

DATALOADER_CONFIG = {'batch_size': 2, 'shuffle': True}

NUM_FEATURES = 18
NUM_CLASSES = 4

WEIGHT_LOSS_FN = True

TRAIN_CONFIG = {
    'loss_fn': torch.nn.CrossEntropyLoss(),  # mutual exclusive classes
    'optim_fn': torch.optim.SGD,
    'lr': 0.001,
    'momentum': 0.9
}

NN_CONFIG = {'act_func': torch.nn.ReLU(),
             'out_func': torch.nn.Softmax(dim=1)}
