import torch
import layers

class Relu(layers.ActivationLayer):

  def describe(self) -> str:
    return __name__
  
  def forward(self, X : torch.Tensor):
    a = torch.max(X, torch.zeros_like(X))
    self.cache['I'] = X
    self.cache['A'] = a
    return a

  def backward(self, prev_grad):
    return torch.where(self.cache['I'] > 0, torch.ones_like(prev_grad), torch.zeros_like(prev_grad)) * prev_grad