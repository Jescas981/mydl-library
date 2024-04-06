from abc import ABC, abstractmethod
import torch
from typing import Dict, Tuple


class Optimizer(ABC):
    @abstractmethod
    def step(self, params_group: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        pass