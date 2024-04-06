from abc import ABC, abstractmethod
from typing import Any
import torch
from contextlib import contextmanager


class Layer(ABC):
    def __init__(self) -> None:
        self.cache = {}

    def describe(self) -> str:
        return __name__

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the activation layer.

        Parameters:
        - X: Input tensor of shape (batch_size, input_size).

        Returns:
        - Y: Output tensor of shape (batch_size, input_size).
        """
        return self.forward(X)

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the activation layer.

        Parameters:
        - X: Input tensor of shape (batch_size, input_size).

        Returns:
        - Y: Output tensor of shape (batch_size, input_size).
        """

    @abstractmethod
    def backward(self, prev_grad: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the activation layer.

        Parameters:
        - prev_grad: Input tensor of previous global gradient.

        Returns:
        - grad: Output tensor of actual global gradient.
        """


class ActivationLayer(Layer):
    def __init__(self) -> None:
        super().__init__()


class ParameterLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.params = {}
        self.grads = {}
