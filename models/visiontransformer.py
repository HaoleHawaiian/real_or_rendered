import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor, logging
logging.set_verbosity_error()
from data_loader import get_train_test_loaders
import matplotlib.pyplot as plt


class ProjectVisionTransformer:

    def __init__(self, epochs=1, learning_rate=2e-5, batch_size=16, optimizer='SGD', momentum=0.9, weight_decay=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = 2
        self.momentum = momentum
        self.weight_decay = weight_decay


        self.model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-distilled-patch16-224', num_labels=self.num_classes).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')

        self.criterion = nn.CrossEntropyLoss()

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.train_loader = None
        self.test_loader = None
        self.attack_loader = None
        self.train_set = None
        self.test_set = None
        self.attack_set = None

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0

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
            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=images).logits
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                progress_bar.refresh()
            
            # Viz prep - training loss
            epoch_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(epoch_train_loss)
            
            # Viz prep - validation loss
            if self.test_loader is not None:
                val_loss = self.evaluate_loss()
                self.val_losses.append(val_loss)

        # Saving model
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            print(f"Created directory: {'saved_models'}")

        if description == '':
            full_save_path = os.path.join('saved_models', f'visiontransformer.pth')
        else:
            full_save_path = os.path.join('saved_models', f'visiontransformer_{description}.pth')

        torch.save(self.model.state_dict(), full_save_path)
        print(f'Model state dictionary saved to: {full_save_path}')

    def model_load(self, file='saved_models/visiontransformer.pth', description=''):
        if description == '':
            self.model.load_state_dict(torch.load(file))
        else:
            description_file = f'saved_models/visiontransformer_{description}.pth'
            self.model.load_state_dict(torch.load(description_file))

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        num_human = 0
        num_ai = 0
        incorrect_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images).logits
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
                            #'path': paths[i],
                            'true_label': labels[i].item(),
                            'predicted_label': predicted[i].item(),
                            'confidence': probs[i][predicted[i]].item()
                        })

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return correct, total, incorrect_preds
    
    def evaluate_loss(self):
        self.model.eval()
        total_loss = 0.0
    
        with torch.no_grad():
            for images, labels, paths in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(pixel_values = images).logits
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
    
        avg_val_loss = total_loss / len(self.test_loader)
        self.model.train()
        return avg_val_loss

    def plot_losses(self):
        # Training
        plt.figure(figsize = (8,6))
        plt.plot(self.train_losses, marker = 'o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        
        # Validation
        plt.figure(figsize = (8,6))
        plt.plot(self.val_losses, marker='x', color='orange')
        plt.title('Validation Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()