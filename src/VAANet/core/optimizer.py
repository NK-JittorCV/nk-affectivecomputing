import jittor.nn as nn
from jittor.optim import Adam


def get_optim(opt, parameters):
    optimizer = Adam(list(filter(lambda p: p.requires_grad, parameters)),
                     lr=opt.learning_rate,
                     weight_decay=opt.weight_decay)
    return optimizer
