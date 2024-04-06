from utilities import DataLoader
from functional import StandardScaler
from models import MNIST64
import torch

if __name__ == "__main__":
    # Load dataset
    base_url = 'http://yann.lecun.com/exdb/mnist'
    loader = DataLoader(download_dir='/home/jescas/mydl_framework/data',
                        urls=[
                            f"{base_url}/train-images-idx3-ubyte.gz",
                            f"{base_url}/train-labels-idx1-ubyte.gz",
                            f"{base_url}/t10k-images-idx3-ubyte.gz",
                            f"{base_url}/t10k-labels-idx1-ubyte.gz"
                        ])

    train_images, train_labels, test_images, test_labels = loader.read_dataset()

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

    # Training proccess
    epochs = 1
    for epoch in range(epochs):
        mnist_model(X_train)
