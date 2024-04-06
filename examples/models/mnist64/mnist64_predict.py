import torch
import MNIST64

mnist_model = torch.load('mnist_model.pth')
params = torch.load('mnist_params.pth')

# Check if the loaded object is an instance of model.MNIST64
if isinstance(mnist_model, MNIST64):
    mnist_model = mnist_model
else:
    raise TypeError("The loaded object is not an instance of model.MNIST64")

mnist_model.update_params(params)