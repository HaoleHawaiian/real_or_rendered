import os
import pandas as pd
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from data_augmentation import get_train_transform, get_test_transform
from data_attack import get_attack_transform


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
        return image, label, img_path

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f)
                            for f in os.listdir(image_folder)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]

def get_dataloader(csv_path='data/train.csv', image_folder='data/train_data', image_size=(224, 224), batch_size=32, shuffle=True):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = CustomImageDataset(csv_path, image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return dataloader, dataset


def get_train_test_loaders(csv_path='data/train.csv', image_folder='data/train_data', image_size=(224, 224), batch_size=32, split_ratio=0.8, augmentation=True, attack_style='SaltAndPepper'):
    #attack style options: SaltAndPepper, Posterize, GaussNoise, RandomShadow
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = CustomImageDataset(csv_path, image_folder, transform=None)

    # To augment or not to augment, that is the question
    if augmentation:
        train_transform = get_train_transform(image_size)
        test_transform = get_test_transform(image_size)
        attack_transform = get_attack_transform(image_size, attack_style='SaltAndPepper')
    else:
        # Use the transformation already written
        train_transform = transform
        test_transform = transform
        attack_transform = get_attack_transform(image_size, attack_style='SaltAndPepper')

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
    train_full_dataset = CustomImageDataset(csv_path, image_folder, transform=train_transform)
    test_full_dataset = CustomImageDataset(csv_path, image_folder, transform=test_transform)
    attack_full_dataset = CustomImageDataset(csv_path, image_folder, transform=attack_transform)

    # Wrap in Subset to use the indices
    train_dataset = Subset(train_full_dataset, train_indices)
    test_dataset = Subset(test_full_dataset, test_indices)
    attack_dataset = Subset(attack_full_dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    attack_loader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader, attack_loader, train_dataset, test_dataset, attack_dataset

def get_unlabeled_loader(image_folder='data/test_data_v2', image_size=(224, 224), batch_size=32):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = UnlabeledImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return dataloader, dataset

if __name__ == "__main__":
    csv_path = 'data/train.csv'
    image_folder = 'data/train_data'

    train_loader, test_loader, attack_loader, train_dataset, test_dataset, attack_dataset = get_train_test_loaders(csv_path, image_folder, attack_style='SaltAndPepper')

    for images, labels, _ in train_loader:
        print("Train batch shape:", images.shape)
        print("Train labels shape:", labels.shape)
        break

    for images, labels, _ in test_loader:
        print("Test batch shape:", images.shape)
        print("Test labels shape:", labels.shape)
        break
    
    for images, labels, _ in attack_loader:
        print("Attack test batch shape:", images.shape)
        print("Attack test labels shape:", labels.shape)
        break
    print(f"Total images in train set: {len(train_dataset)}")
    print(f"Total images in test set: {len(test_dataset)}")
    print(f"Total images in attack test set: {len(attack_dataset)}")
