import torch
from ..functional import ParamInit
from . import ParameterLayer


class Conv(ParameterLayer):
    def __init__(self, size):
        super().__init__()
        self.params['k'] = ParamInit()(size, 'conv')

    def describe(self) -> str:
        return f"{__name__}{list(self.params['k'].shape)}"

    def forward(self, X: torch.Tensor):
        self.cache['I'] = X
        kernel = self.params['k']
        k = kernel.shape[1]
        m, n, d = X.shape
        output = torch.empty(m, n - k + 1, d - k + 1)
        for i in range(n-k+1):
            for j in range(d-k+1):
                w = X[:, i:i+k, j:j+k]  # Window samples
                output[:, i, j] = torch.einsum('ijk,jk->i', w, kernel)
        return output

    def backward(self, prev_grad: torch.Tensor):
        # Conv multiple channels
        I = self.cache['I']
        k = prev_grad.shape[1]
        m, n, d = I.shape
        dk = torch.empty(m, n - k + 1, d - k + 1)

        # Convolution of dk
        for i in range(n-k+1):
            for j in range(d-k+1):
                w = I[:, i:i+k, j:j+k]  # Window samples
                dk[:, i, j] = torch.einsum('ijk,ijk->i', w, prev_grad)

        self.grads['k'] = torch.mean(dk, dim=0)

        # Conv multiple channels
        p = int((I.shape[1] - 2) / 2)  # Compute padding
        kernel_pad = torch.nn.functional.pad(self.params['k'], (p, p, p, p))
        prev_grad_rot = torch.rot90(prev_grad, k=2, dims=[1, 2])
        m, _, k = prev_grad.shape
        n, d = kernel_pad.shape
        do = torch.empty(m, n - k + 1, d - k + 1)

        # Convolution of do
        for i in range(n-k+1):
            for j in range(d-k+1):
                w = kernel_pad[i:i+k, j:j+k]  # Window samples
                do[:, i, j] = torch.einsum('jk,ijk->i', w, prev_grad_rot)

        return do
