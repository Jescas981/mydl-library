import torch
from ..functional import ParamInit
from . import ParameterLayer


class Linear(ParameterLayer):

    def __init__(self, size, device: str = "cpu"):
        super().__init__()
        self.params['w'] = ParamInit()(size, 'kaiming', device)
        self.params['b'] = ParamInit()((1, size[1]), 'zero', device)

    def describe(self) -> str:
        return f"{__name__} | w{list(self.params['w'].shape)} - b{list(self.params['b'].shape)}"

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.cache['I'] = X
        a = X@self.params['w'] + self.params['b']
        self.cache['A'] = a
        return a

    def backward(self, prev_grad: torch.Tensor) -> torch.Tensor:
        di = prev_grad@self.params['w'].T
        self.grads['w'] = (self.cache['I'].T@prev_grad) / di.shape[0]
        self.grads['b'] = torch.mean(prev_grad, dim=0, keepdim=True)
        return di
