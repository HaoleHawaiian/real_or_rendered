import os
import pandas as pd
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from data_augmentation import get_train_transform, get_test_transform



class CustomImageDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, csv_path, image_folder, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['label'].isin([0, 1])]
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file_name = os.path.basename(self.img_labels.loc[idx, 'file_name'])
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.loc[idx, 'label'])
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(csv_path='data/train.csv', image_folder='data/train_data', image_size=(224, 224), batch_size=32, shuffle=True):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    dataset = CustomImageDataset(csv_path, image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

def get_train_test_loaders(csv_path='data/train.csv', image_folder='data/train_data', image_size=(224, 224), batch_size=32, split_ratio=0.8, augmentation=True):
    """
    
    """
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    dataset = CustomImageDataset(csv_path, image_folder, transform=None)

    # To augment or not to augment, that is the question
    if augmentation:
        train_transform = get_train_transform(image_size)
        test_transform = get_test_transform(image_size)
    else:
        # Use the transformation already written
        train_transform = transform
        test_transform = transform

    # Split the data into train and test
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)                  # randomize the split
    train_size = int(split_ratio * len(dataset))
    # test_size = len(dataset) - train_size
    
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    
    # Create subsets with transform
    train_dataset = CustomImageDataset(csv_path, image_folder, transform=train_transform)
    test_dataset = CustomImageDataset(csv_path, image_folder, transform=test_transform)

    # Wrap in Subset to use the indices
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    csv_path = 'data/train.csv'
    image_folder = 'data/train_data'

    train_loader, test_loader, train_set, test_set = get_train_test_loaders(csv_path, image_folder)

    for images, labels in train_loader:
        print("Train batch shape:", images.shape)
        print("Train labels shape:", labels.shape)
        break

    for images, labels in test_loader:
        print("Test batch shape:", images.shape)
        print("Test labels shape:", labels.shape)
        break

    print(f"Total images in train set: {len(train_set)}")
    print(f"Total images in test set: {len(test_set)}")
