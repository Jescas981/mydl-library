from . import Optimizer
from typing import Dict, Tuple, Generator
import torch


class BatchGradientDescent(Optimizer):
    def __init__(self, lr=1e-3) -> None:
        self.lr = lr

    def step(self, params_group: Generator[Dict[str, Tuple[torch.Tensor, torch.Tensor]], None, None]):
        for group in params_group:
            for key, (param, grad) in group.items():
                param.data -= self.lr * grad.data
