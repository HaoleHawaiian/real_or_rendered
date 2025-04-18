# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:33:17 2025

@author: aloha
"""
# Machine Learning
import torch
from transformers import ViTForImageClassification, ViTImageProcessor 
# https://huggingface.co/docs/transformers/en/model_doc/vit
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/vit/modeling_vit.py#L780
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/vit/image_processing_vit.py#L42
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision import transforms

# Viz, analysis
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
import seaborn as sns
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
from tqdm import tqdm
# https://tqdm.github.io/

# data read class
from data_reader import RealOrRenderedDataset


class ViTTrainer:
    def __init__(self, csv_file, root_dir, save_path = "best_vit_model.pth", num_epochs = 5, batch_size = 64, val_split = 0.2, lr = 2e-5, weight_decay = 0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.save_path = save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Call the pretrained models
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels = 2).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Resize/Preprocess imagees
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.processor.image_mean, std = self.processor.image_std)
        ])
        # https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html

        full_dataset = RealOrRenderedDataset(csv_file = csv_file, root_dir = root_dir, transform = transform)
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        
        # Results at the end
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Save for initial best model, gets overwritten if the current model's accuracy is higher than the most recent
        self.best_val_acc = 0.0

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            print(f"\nEpoch {epoch + 1} starting...")

            # Anaconda prompt output description
            for inputs, labels in tqdm(self.train_loader, desc = f"Epoch {epoch+1} - Training"):
                
                # Forward Pass
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values = inputs).logits
                loss = self.criterion(outputs, labels)

                # Backward Pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            # Training visualizations and results -- NOT NEEDED FOR CALCULATIONS
            avg_train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}")

            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved with accuracy: {val_acc * 100:.2f}%")

        self.plot_curves()

    # For evaluation at the end of the training -- NOT NEEDED FOR CALCULATIONS
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc = "Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values = inputs).logits
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = correct / total
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc * 100:.2f}%")
        return avg_val_loss, val_acc

    # Visualizations -- NOT NEEDED FOR CALCULATIONS
    # def plot_curves(self):
    #     plt.figure(figsize = (10, 4))

    #     # Loss curve
    #     plt.subplot(1, 2, 1)
    #     plt.plot(self.train_losses, label = "Train Loss")
    #     plt.plot(self.val_losses, label = "Val Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.title("Loss Curve")
    #     plt.legend()

    #     # Accuracy curve
    #     plt.subplot(1, 2, 2)
    #     plt.plot([acc * 100 for acc in self.val_accuracies], label = "Val Accuracy", color = 'green')
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Accuracy (%)")
    #     plt.title("Validation Accuracy")
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.savefig("training_curves.png")
    #     plt.show()

    def evaluate(self):
        # Load best model
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc = "Evaluating Best Model"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values = inputs).logits
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize = (6, 5))
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                    xticklabels = ['Real', 'Rendered'], yticklabels = ['Real', 'Rendered'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

        # Classification Report
        report = classification_report(all_labels, all_preds, target_names = ['Real', 'Rendered'], digits = 2)
        print("Classification Report:\n")
        print(report)

        with open("classification_report.txt", "w") as f:
            f.write(report)

if __name__ == "__main__":
    trainer = ViTTrainer(
        csv_file ='data/train.csv',
        root_dir ='data',
        save_path ='best_vit_model.pth',
        # Tuning here!
        num_epochs = 2,
        batch_size = 64,
        lr = 2e-5,
        weight_decay = 0.01        
    )
    trainer.train()
    trainer.evaluate()
