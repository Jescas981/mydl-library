from utilities import DataLoader
from optimizers import BatchGradientDescent
from functional import StandardScaler
from models import MNIST64
import torch
from sklearn.datasets import load_digits
from losses import CategoricalCrossEntropyLoss
from sklearn.model_selection import train_test_split
from functional.metrics import accuracy

if __name__ == "__main__":
    digits = load_digits()
    # Split the dataset into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42)
    train_images = train_images.reshape(-1, 8, 8)
    test_images = test_images.reshape(-1, 8, 8)

    # Normalize data
    scaler = StandardScaler()
    train_images_norm = scaler.fit_transform(train_images)
    test_images_norm = scaler.transform(test_images)

    # Turn data to tensors
    X_train = torch.tensor(train_images_norm, dtype=torch.float32)
    X_test = torch.tensor(test_images_norm, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    mnist_model = MNIST64()
    loss_fn = CategoricalCrossEntropyLoss()
    optim = BatchGradientDescent(lr=1e-1)

    # Training proccess
    epochs = 1000
    loss_h = []
    tloss_h = []
    for epoch in range(epochs):
        # Forward propagation
        y_pred = mnist_model(X_train)

        if epoch % 50 == 0:
            # Metrics
            loss = loss_fn(y_true=y_train, y_hat=y_pred)
            acc = accuracy(y_train, y_pred)*100
            # Test validation
            with mnist_model.cache_preserve():
                y_pred_test = mnist_model(X_test)
                tloss = loss_fn(y_true=y_test, y_hat=y_pred_test)
                tacc = accuracy(y_test, y_pred_test)*100

            print(
                f"[Epoch: {epoch}/{epochs}] - Loss: {loss.item():.6f} - Acc: {acc:.2f}%"
                f" | TestLoss: {tloss.item():.6f} - TestAcc: {tacc:.2f}% ")

        # Backward propagation
        loss_grad = loss_fn.backward(y_true=y_train, y_hat=y_pred)
        mnist_model.backward(loss_grad)
        # Update parameters
        optim.step(mnist_model.param_groups())

    print(mnist_model.describe())
    # Save model parameters
    torch.save(mnist_model.state_dict(), 'mnist_params.pth')
    torch.save(mnist_model, 'mnist_model.pth')
