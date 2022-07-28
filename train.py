import torch
import os
from config import DATALOADER_CONFIG, NUM_EPOCHS, TRAIN_CONFIG
from torch.utils.data import DataLoader
from ZapfenDataset import ZapfenDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train(model):

    dataset = ZapfenDataset('./zapfen.csv')
    trainset, testset = dataset.get_train_and_test_set(0.8)
    trainloader = DataLoader(trainset, **DATALOADER_CONFIG)
    testloader = DataLoader(testset, **DATALOADER_CONFIG)
    writer = SummaryWriter(log_dir=os.path.join('logging', 'tensorboard', datetime.now()))

    loss_fn = TRAIN_CONFIG['loss_fn']
    batch_size = DATALOADER_CONFIG['batch_size']
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_processed, num_correct = 0, 0
    print(f'Number of model parameters {model.get_num_params()}')
    print(f'Number of samples {len(trainloader) * batch_size}')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for i, (batch_features, batch_labels) in enumerate(trainloader):

            # run model
            optimizer.zero_grad()
            batch_prediction = model(batch_features)

            # calc batch accuracy
            batch_acc = 0
            for (prediction, labels) in zip(batch_prediction, batch_labels):
                batch_acc += get_acc(prediction, labels)
            num_correct += batch_acc
            batch_acc /= batch_size * 100

            loss = loss_fn(batch_prediction, batch_labels)
            writer.add_scalar('loss/train', loss.item(), num_processed)
            writer.add_scalar('accuracy/train', batch_acc, num_processed)

            num_processed += batch_size

            if i % 30 == 0:
                print(f'Step {i}/{len(trainloader)}. Epoch {epoch + 1}.')

    # Training Summary
    writer.add_text('', f'{num_correct / num_processed * 100}% accuracy on trainset.')
    print(f'{num_correct / num_processed * 100}% accuracy on trainset.')

    num_processed, num_correct = 0, 0
    # Validation Loop
    with torch.no_grad():
        for batch_features, batch_labels in testloader:
            # run model
            batch_prediction = model(batch_features)

            # calc batch accuracy
            batch_acc = 0
            for (prediction, labels) in zip(batch_prediction, batch_labels):
                batch_acc += get_acc(prediction, labels)
            num_correct += batch_acc
            batch_acc /= batch_size

            writer.add_scalar('accuracy/test', batch_acc, num_processed)
            num_processed += batch_size

    # Validation Summary
    writer.add_text('', f'{num_correct / num_processed * 100}% accuracy on testset.')
    print(f'{num_correct / num_processed * 100}% accuracy on testset.')


def get_acc(prediction, label):
    """ Checks whether prediction with highest probability is
        the correct class.

    Args:
        prediction (tensor): predicted class probabilities
        label (tensor): actual target classes.

    Returns:
        int: 1 if correct class 0 otherwise.
    """
    prediction, label = prediction.tolist(), label.tolist()
    idx_pred = prediction.index(max(prediction))
    idx_lab = label.index(1.0)
    if idx_pred == idx_lab:
        return 1
    else:
        return 0
