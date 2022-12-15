from ZapfenDataset import ZapfenDataset
from models.ffnn import FFNN1, FFNN2, FFNN3, FFNN4
from train import train
from torch.nn import ReLU, LeakyReLU, CrossEntropyLoss
from torch import tensor
from config import TRAIN_CONFIG, NN_CONFIG, DATALOADER_CONFIG, WEIGHT_LOSS_FN
import torch
import sys


def launch():
    # TODO: train all with same weight initialisation
    global WEIGHT_LOSS_FN
    for batch_size in [4, 1, 2, 4, 8, 16]:
        for act_func in [ReLU(), LeakyReLU()]:
            for apply_weight in [True, False]:
                for nn in [FFNN1(), FFNN2(), FFNN3(), FFNN4()]:
                    for lr in [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03]:
                        for mom in [0.7, 0.75, 0.8, 0.85, 0.9]:
                            if apply_weight:
                                loss_fn = CrossEntropyLoss(weight=tensor([0.28, 0.1, 0.1, 0.52]))
                            else:
                                loss_fn = CrossEntropyLoss()
                            DATALOADER_CONFIG['batch_size'] = batch_size
                            TRAIN_CONFIG['loss_fn'] = loss_fn
                            TRAIN_CONFIG['lr'] = lr
                            TRAIN_CONFIG['momentum'] = mom
                            NN_CONFIG[act_func] = act_func
                            WEIGHT_LOSS_FN = apply_weight
                            if WEIGHT_LOSS_FN:
                                weight = 'weight_applied'
                            else:
                                weight = ''
                            context_str = str(batch_size) + '_' + str(loss_fn) + str(act_func) + weight
                            context_str = context_str.replace(')', '_').replace('(', '_')
                            train(nn, context_str)
                            sys.exit()
    return
    ds = ZapfenDataset('./zapfen.csv')
    ds.plot_label_distribution()
    return
    ds.plot_feature_distribution(fname='feature_distr_before_fixing', title='before_fixing')
    ds.fix_invalid_values()
    ds.plot_feature_distribution(fname='feature_distr_after_fixing', title='after_fixing')
    ds.scale()
    ds.plot_feature_distribution(fname='after_fixing_and_normalizing', title='after_fixing_and_normalizing')
    ds.plot_label_distribution(fname='label_distr_after')


if __name__ == '__main__':
    launch()
