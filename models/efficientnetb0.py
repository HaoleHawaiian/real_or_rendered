import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F
from data_loader import get_train_test_loaders

class ProjectEfficientNet:

    def __init__(self, epochs=5, learning_rate=0.001, batch_size=16, optimizer='SGD', momentum=0.9, weight_decay=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = 2
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.momentum = momentum
        self.weight_decay = weight_decay

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.train_loader = None
        self.test_loader = None
        self.attack_loader = None
        self.train_set = None
        self.test_set = None
        self.attack_set = None

    def data_load(self, train_loader=None, test_loader=None):
        if train_loader is not None:
            self.train_loader = train_loader
        if test_loader is not None:
            self.test_loader = test_loader

    def train(self, description=''):
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}', leave=False, unit='batch')
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
                progress_bar.refresh()

        # Saving model
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            print(f"Created directory: {'saved_models'}")

        full_save_path = os.path.join('saved_models', f'efficientnet_b0_model_{description}.pth')

        torch.save(self.model.state_dict(), full_save_path)
        print(f'Model state dictionary saved to: {full_save_path}')

    def model_load(self, file='saved_models/efficientnet_b0_model.pth', description=''):
        if description == '':
            self.model.load_state_dict(torch.load(file))
        else:
            description_file = f'saved_models/efficientnet_b0_model_{description}.pth'
            self.model.load_state_dict(torch.load(description_file))

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        num_human = 0
        num_ai = 0
        incorrect_preds = []

        with torch.no_grad():
            for images, labels, paths in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    if labels[i] == 0:
                        num_human += 1
                    elif labels[i] == 1:
                        num_ai += 1

                    if predicted[i] != labels[i]:
                        incorrect_preds.append({
                            'path': paths[i],
                            'true_label': labels[i].item(),
                            'predicted_label': predicted[i].item(),
                            'confidence': probs[i][predicted[i]].item()
                        })

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return correct, total, incorrect_preds