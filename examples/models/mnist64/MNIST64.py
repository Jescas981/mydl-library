from mydllib import layers, modules
from typing import List
import torch


class MNIST64(modules.Model):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)

        self.layers: List[layers.Layer] = [
            layers.Conv(size=(5, 5)),
            layers.Conv(size=(3, 3)),
            layers.Flatten(dim=1),
            layers.Linear(size=(4, 10)),
            layers.Relu(),
            layers.Linear(size=(10, 10)),
            layers.Softmax(n_classes=10),
        ]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        a = X
        for l in self.layers:
            a = l(a)
        return a

    def backward(self, loss_grad: torch.Tensor):
        grad = loss_grad
        for l in reversed(self.layers):
            grad = l.backward(grad)
