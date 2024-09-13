from dataclasses import dataclass, field

#Objects beginning with _ won't be imported when: from config import *

_Optimizer = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'weight_decay': 0.0000
}

_Dataset = {
    "n": 10000
}

_Train = {
    "batch": 32,
    "epochs": 500
}

_NN = {
    "w": 128
}


_FM = {
    "sigma_min": 0.001,
    "t_dist": "power_law",
    "alpha": -0.2
}

@dataclass
class Configuration:
    Optimizer: dict = field(default_factory=lambda: _Optimizer)
    Dataset: dict = field(default_factory=lambda: _Dataset)
    Train: dict = field(default_factory=lambda: _Train)
    FM: dict = field(default_factory=lambda: _FM)
    NN: dict = field(default_factory=lambda: _NN)


@dataclass
class Diagnostics:
    losses: list = field(default_factory=list)
    epoch_tr_loss: list = field(default_factory=list)
    epoch_val_loss: list = field(default_factory=list)
    epoch_val_accuracy: list = field(default_factory=list)

