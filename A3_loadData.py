import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') #Set GUI backend for plots

def compute_mean_std(dataset):
    """Compute mean and std of the dataset"""
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

#load raw training dataset (without normalization) to compute stats
temp_train_dataset = datasets.FashionMNIST(root='./data',train=True,download=True,transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
train_mean, train_std = compute_mean_std(temp_train_dataset)

#define transforms with normalization using training stats
train_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=train_mean, std=train_std),
    #uncomment for data augmentation
    #v2.RandomApply(transforms=[v2.RandomCrop(size=(28, 28))], p=0.5),
    #v2.RandomHorizontalFlip(p=0.5),
    #v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 180))], p=0.5)
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=train_mean, std=train_std)
])

#load full datasets with transforms
full_train_dataset = datasets.FashionMNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=train_transform
)

test_dataset = datasets.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=test_transform
)

#split training data into train and validation
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f"Train set size: {len(train_dataset)}")
print(f"Val set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

#TODO make bar chart of class distributions
#visualize some samples
labels_map = {
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

figure = plt.figure(figsize=(10, 10))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    image, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap='gray')

plt.show()
plt.close()

#convert to numpy for MLP training (flatten images)
def dataset_to_numpy(dataset):
    """convert PyTorch dataset to numpy arrays"""
    images_list = []
    labels_list = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    for images, labels in loader:
        #flatten images from (batch, 1, 28, 28) to (batch, 784)
        images = images.view(images.size(0), -1)
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())
    
    X = np.concatenate(images_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

X_train, y_train = dataset_to_numpy(train_dataset)
X_val, y_val = dataset_to_numpy(val_dataset)
X_test, y_test = dataset_to_numpy(test_dataset)

#save numpy arrays for later use
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
