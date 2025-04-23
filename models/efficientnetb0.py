import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from skimage.exposure import match_histograms

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F

from data_loader import get_train_test_loaders

class ProjectEfficientNet:

    def __init__(self, epochs=5, learning_rate=0.001, batch_size=16, num_classes=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_loader = None
        self.test_loader = None
        self.attack_loader = None
        self.train_set = None
        self.test_set = None
        self.attack_set = None

    def data_load(self, split_ratio=0.8):
        train_loader, test_loader, attack_loader, train_set, test_set, attack_set = get_train_test_loaders(split_ratio=0.8)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.attack_loader = attack_loader
        self.train_set = train_set
        self.test_set = test_set
        self.attack_set = attack_set

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}', leave=True, unit='batch')
            for i, (images, labels, path) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # --- Save the Model ---
        # Create the directory if it doesn't exist
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            print(f"Created directory: {'saved_models'}")

        # Construct the full path for saving the model
        full_save_path = os.path.join('saved_models', f'efficientnet_b0_model_{self.epochs}epochs.pth')

        # Save the model's state dictionary
        torch.save(self.model.state_dict(), full_save_path)
        print(f'Model state dictionary saved to: {full_save_path}')

    def model_load(self):

    def evaluate(self):




def train_efficientnet():


