import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

TRANSFORM = ToTensor()

# first load the train + validation datasets
# dont forget to transform the data to tensor

train_data = torchvision.datasets.MNIST(   # image, label = mnist_data[0]
    root="./",
    train=True,
    download=True,
    transform=TRANSFORM
)

validation_data = torchvision.datasets.MNIST(   # image, label = mnist_data[0]
    root="./",
    train=False,
    download=True,
    transform=TRANSFORM
)


# create dataloaders

train_loader = DataLoader(train_data, batch_size=32) # all the data but split into batches of 32
val_loader = DataLoader(validation_data, batch_size=32)