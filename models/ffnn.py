import torch.nn as nn
from functools import reduce
from config import NUM_FEATURES, NUM_CLASSES, NN_CONFIG


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.l1 = nn.Linear(NUM_FEATURES, 11)
        self.l2 = nn.Linear(11, 11)
        self.l3 = nn.Linear(11, NUM_CLASSES)
        self.act_func = NN_CONFIG['act_func']
        self.out_func = NN_CONFIG['out_func']

    def forward(self, x):
        out = self.act_func(self.l1(x))
        out = self.act_func(self.l2(out))
        out = self.out_func(self.l3(out))
        return out

    def get_num_params(self):
        "count number of trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in self.parameters())
        return total_params
