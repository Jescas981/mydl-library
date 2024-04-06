from abc import ABC, abstractmethod
from typing import Any
import torch


class Loss(ABC):
    def __call__(self,  y_true: torch.Tensor, y_hat: torch.Tensor) -> Any:
        """
        Forward pass of the loss function

        Parameters:
        - y_true: Input tensor of shape (batch_size, input_size).
        - y_hat: Input tensor of shape (batch_size, input_size).

        Returns:
        - l: Output tensor of shape (batch_size, input_size).
        """
        return self.forward(y_true, y_hat)

    @abstractmethod
    def forward(self, y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function

        Parameters:
        - y_true: Input tensor of shape (batch_size, input_size).
        - y_hat: Input tensor of shape (batch_size, input_size).

        Returns:
        - l: Output tensor of shape (batch_size, input_size).
        """

    @abstractmethod
    def backward(self, y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the loss function.

        Parameters:
        - prev_grad: Input tensor of previous global gradient.

        Returns:
        - grad: Output tensor of actual global gradient.
        """
