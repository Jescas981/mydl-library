import torch
import layers


class Softmax(layers.ActivationLayer):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def describe(self) -> str:
        return __name__

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _, max = torch.max(X, dim=1, keepdim=True)
        exp = torch.exp(X-max)
        a = exp/torch.sum(exp, dim=1, keepdim=True)
        self.cache['I'] = X
        self.cache['A'] = a
        return a

    def backward(self, prev_grad: torch.Tensor) -> torch.Tensor:
        s = self.cache['A']
        J = -s.unsqueeze(1) * s.unsqueeze(2)
        xx, yy = torch.diag_embed(torch.ones(
            J.size(1))).byte().nonzero(as_tuple=True)
        J[:, xx, yy] = s * (1 - s)
        grad = torch.einsum('ijk,ik->ij', J, prev_grad)
        return grad
