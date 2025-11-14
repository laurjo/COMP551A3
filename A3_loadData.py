import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=32)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        # Mean over batch, height, and width, for each channel
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt(channels_squared_sum / num_batches - mean**2)

    return mean, std

#load training dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
train_mean, train_std = compute_mean_std(train_dataset)

#load testing dataset
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_mean, test_std = compute_mean_std(test_dataset)

#define transforms for datasets to be used in training
train_transform = v2.Compose([
                #change to tensor image
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=train_mean, std=train_std),
                #crop with probaility 0.5
                v2.RandomApply(transforms=[v2.RandomCrop(size=(28, 28))], p=0.5),
                #flip with probability 0.5
                v2.RandomHorizontalFlip(p=0.5),
                #rotate with probaility 0.5
                v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 180))], p=0.5)
                ])

#define transforms for datasets to be used in testing
test_transform = train_transform = v2.Compose([
                #change to tensor image
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=test_mean, std=test_std)
                ])

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=train_transform)
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=test_transform)


#TODO: come back and fix this so that I get one image from each class, make bar chart of class distributions
labels_map={
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}
print(labels)
figure = plt.figure(figsize = (10,10))
cols, rows = 3, 3

for i in range (1, cols*rows + 1):
    sample_idx = torch.randint(len(train_dataset), size = (1,)).item()
    image, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap='gray')
plt.show()