import torch
from contextlib import contextmanager
import layers
from typing import Dict, Tuple, Generator, List
from abc import ABC, abstractmethod


class Model(ABC):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    @contextmanager
    def cache_preserve(self):
        cache = {}
        for i, layer in enumerate(self.layers, 1):
            cache[f'l{i}'] = layer.cache.copy()
        try:
            yield
        finally:
            for i, layer in enumerate(self.layers, 1):
                layer.cache = cache[f'l{i}']

    def describe(self) -> str:
        description_str = ""
        description_str += f"### {type(self).__name__} ###\n"
        for i, layer in enumerate(self.layers, 1):
            description_str += f"{i}: {layer.describe()}\n"
        return description_str

    def update_params(self, state: Dict[str, Dict[str, torch.Tensor]]):
        for key, group in state.items():
            l = int(key.split('_')[0][1:])  # Extract and convert to int
            self.layers[l].params = group

    def state_dict(self):
        state = {}
        for i, layer in enumerate(self.layers, 1):
            if hasattr(layer, 'params'):
                state[f'l{i}_{type(layer).__name__}'] = layer.params.copy()
        return state

    def param_groups(self) -> Generator[Dict[str, Tuple[torch.Tensor, torch.Tensor]], None, None]:
        for i, layer in enumerate(self.layers, 1):
            if hasattr(layer, 'params'):
                yield {f'l{i}_{type(layer).__name__}_{key}': (layer.params[key], layer.grads[key]) for key in layer.params.keys()}

    @ abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @ abstractmethod
    def backward(self, loss_grad: torch.Tensor) -> torch.Tensor:
        pass
