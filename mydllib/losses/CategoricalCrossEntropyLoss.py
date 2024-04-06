from mydllib.losses import Loss
import torch
from functional.one_hot import one_hot

class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-32
        y_pred = torch.clip(y_hat, epsilon, 1 - epsilon)
        y_target = one_hot(y_true, n_features=y_hat.shape[1])
        return -torch.sum(y_target * torch.log(y_pred)) / y_true.shape[0]

    def backward(self, y_true: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-32
        y_pred = torch.clip(y_hat, epsilon, 1 - epsilon)
        y_target = one_hot(y_true, n_features=y_hat.shape[1])
        return - y_target / y_pred
