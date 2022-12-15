import torch.nn as nn
from functools import reduce
from config import NUM_FEATURES, NUM_CLASSES, NN_CONFIG


class FFNN1(nn.Module):
    def __init__(self):
        super(FFNN1, self).__init__()
        self.name = 'potatoe'
        self.l1 = nn.Linear(NUM_FEATURES, 30)
        self.l2 = nn.Linear(30, 20)
        self.l3 = nn.Linear(20, 10)
        self.l4 = nn.Linear(10, NUM_CLASSES)
        self.act_func = NN_CONFIG['act_func']
        self.out_func = NN_CONFIG['out_func']

    def forward(self, x):
        out = self.act_func(self.l1(x))
        out = self.act_func(self.l2(out))
        out = self.out_func(self.l3(out))
        out = self.act_func(self.l4(out))
        return out

    def get_num_params(self):
        "count number of trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in self.parameters())
        return total_params


class FFNN2(nn.Module):
    def __init__(self):
        super(FFNN2, self).__init__()
        self.name = 'long_but_steady'
        self.l1 = nn.Linear(NUM_FEATURES, NUM_FEATURES)
        self.l2 = nn.Linear(NUM_FEATURES, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, 8)
        self.l5 = nn.Linear(8, NUM_CLASSES)
        self.act_func = NN_CONFIG['act_func']
        self.out_func = NN_CONFIG['out_func']

    def forward(self, x):
        out = self.act_func(self.l1(x))
        out = self.act_func(self.l2(out))
        out = self.out_func(self.l3(out))
        out = self.act_func(self.l4(out))
        out = self.act_func(self.l5(out))
        return out

    def get_num_params(self):
        "count number of trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in self.parameters())
        return total_params


class FFNN3(nn.Module):
    def __init__(self):
        super(FFNN3, self).__init__()
        self.name = 'get_big'
        self.l1 = nn.Linear(NUM_FEATURES, 25)
        self.l2 = nn.Linear(25, 40)
        self.l3 = nn.Linear(40, 40)
        self.l4 = nn.Linear(40, 15)
        self.l5 = nn.Linear(15, 10)
        self.l6 = nn.Linear(10, NUM_CLASSES)
        self.act_func = NN_CONFIG['act_func']
        self.out_func = NN_CONFIG['out_func']

    def forward(self, x):
        out = self.act_func(self.l1(x))
        out = self.act_func(self.l2(out))
        out = self.out_func(self.l3(out))
        out = self.act_func(self.l4(out))
        out = self.act_func(self.l5(out))
        out = self.act_func(self.l6(out))
        return out

    def get_num_params(self):
        "count number of trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in self.parameters())
        return total_params


class FFNN4(nn.Module):
    def __init__(self):
        super(FFNN4, self).__init__()
        self.name = 'small'
        self.l1 = nn.Linear(NUM_FEATURES, 10)
        self.l3 = nn.Linear(10, 7)
        self.l4 = nn.Linear(7, NUM_CLASSES)
        self.act_func = NN_CONFIG['act_func']
        self.out_func = NN_CONFIG['out_func']

    def forward(self, x):
        out = self.act_func(self.l1(x))
        out = self.out_func(self.l3(out))
        out = self.act_func(self.l4(out))
        return out

    def get_num_params(self):
        "count number of trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in self.parameters())
        return total_params
