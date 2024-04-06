import torch


def accuracy(y_true: torch.Tensor, y_hat: torch.Tensor) -> float:
    """
    Computes the accuracy of the predicted values with respect to the true labels.

    Parameters:
    - y_true: True labels (ground truth), tensor of shape (batch_size, num_classes).
    - y_hat: Predicted values, tensor of shape (batch_size, num_classes).

    Returns:
    - acc: Accuracy, a float value between 0.0 and 1.0.
    """
    # Convert predicted values to class labels by taking the argmax
    y_pred_labels = torch.argmax(y_hat, dim=1)
    
    
    # Calculate accuracy
    correct = torch.sum(y_pred_labels == y_true).item()
    total = y_true.size(0)
    acc = correct / total
    
    return acc
