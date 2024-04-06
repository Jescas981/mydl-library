from typing import Any, Tuple
import torch


class ParamInit:
    def __call__(self, size: Tuple[int], method: str) -> torch.Tensor:
        torch.random.manual_seed(10)
        if method == 'zero':
            return torch.zeros(size)
        elif method == 'kaiming':
            torch.random.manual_seed(10)
            return torch.normal(mean=0, std=torch.sqrt(2 / (torch.tensor(size[0]+size[1]))), size=size)
        elif method == 'conv':
            torch.random.manual_seed(10)
            return torch.normal(mean=0, std=torch.sqrt(1 / (torch.tensor(size[0]*size[1]))), size=size)
        else:
            torch.random.manual_seed(10)
            return torch.normal(mean=0, std=torch.sqrt(2 / (torch.tensor(size[0]))), size=size)
