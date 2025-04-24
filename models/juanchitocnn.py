import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_train_test_loaders

class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 12, 4),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(12, 24, 5),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 25 * 25, 2)


    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

class ProjectJuanchitoCNN:

    def __init__(self, epochs=5, learning_rate=0.001, batch_size=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = 2
        self.model = ClassifierModel().to(self.device)
        self.momentum = 0.9

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.train_loader = None
        self.test_loader = None
        self.attack_loader = None
        self.train_set = None
        self.test_set = None
        self.attack_set = None

    def data_load(self, split_ratio=0.8, image_size=(224,224), batch_size=32, augmentation=False):
        train_loader, test_loader, attack_loader, train_set, test_set, attack_set = get_train_test_loaders(split_ratio=split_ratio, image_size=image_size, train_batch_size=batch_size, test_batch_size=batch_size, attack_batch_size=batch_size, augmentation=augmentation)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.attack_loader = attack_loader
        self.train_set = train_set
        self.test_set = test_set
        self.attack_set = attack_set

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}', leave=False, unit='batch')
            for i, (images, labels, path) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # print statistics
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                progress_bar.refresh()

        # Saving model
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            print(f"Created directory: {'saved_models'}")

        full_save_path = os.path.join('saved_models', f'JuanchitoCNN.pth')

        torch.save(self.model.state_dict(), full_save_path)
        print(f'Model state dictionary saved to: {full_save_path}')

    def model_load(self, file='saved_models/JuanchitoCNN.pth'):
        self.model.load_state_dict(torch.load(file))

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