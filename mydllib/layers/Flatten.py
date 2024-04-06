import torch
from . import Layer

class Flatten(Layer):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def describe(self) -> str:
    return f"{__name__}(dim={self.dim})"

  def forward(self, X):
    self.cache['shape'] = X.shape
    return torch.flatten(X, start_dim =self.dim)

  def backward(self, prev_grad):
    shape = self.cache['shape']
    return prev_grad.view(shape)