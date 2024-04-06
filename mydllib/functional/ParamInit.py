from typing import Any, Tuple
import torch


class ParamInit:
    def __call__(self, size: Tuple[int], method: str, device: str = "cpu") -> torch.Tensor:
        if method == 'zero':
            return torch.zeros(size)
        elif method == 'kaiming':
            return torch.normal(mean=0, std=torch.sqrt(2 / (torch.tensor(size[0]+size[1]))), size=size, device=device)
        elif method == 'conv':
            return torch.normal(mean=0, std=torch.sqrt(1 / (torch.tensor(size[0]*size[1]))), size=size, device=device)
        else:
            return torch.normal(mean=0, std=torch.sqrt(2 / (torch.tensor(size[0]))), size=size, device=device)
