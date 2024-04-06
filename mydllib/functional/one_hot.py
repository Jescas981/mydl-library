import torch


def one_hot(y: torch.Tensor, n_features: int) -> torch.Tensor:
    output = torch.zeros(y.shape[0], n_features, device=y.device)
    output[torch.arange(y.shape[0]), y] = 1
    return output
