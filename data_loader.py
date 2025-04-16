import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomImageDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self, csv_path, image_folder, transform=None):
        self.img_labels = pd.read_csv(csv_path, header=None, names=['file_name', 'label'])
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


def get_dataloader(csv_path, image_folder, image_size=(224, 224), batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(csv_path, image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

if __name__ == "__main__":
    csv_path = 'data/train.csv'
    image_folder = 'data/train_data'

    dataloader, dataset = get_dataloader(csv_path, image_folder)

    # Preview one batch
    for images, labels in dataloader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        break

    print(f"Total images in dataset: {len(dataset)}")

