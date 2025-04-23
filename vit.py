# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:33:17 2025

@author: aloha
"""
# Machine Learning
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, logging
logging.set_verbosity_error()
# https://huggingface.co/docs/transformers/en/model_doc/vit
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/vit/modeling_vit.py#L780
# https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/vit/image_processing_vit.py#L42
# from torch.utils.data import DataLoader, random_split
from torch import nn, optim
# from torchvision import transforms

# Viz, analysis
from sklearn.metrics import classification_report
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
from tqdm import tqdm
# https://tqdm.github.io/

# data read class
# from data_reader import RealOrRenderedDataset
from data_loader import get_train_test_loaders

class ViTTrainer:
    def __init__(self, train_loader, test_loader, save_path='best_vit_model.pth', num_epochs=1, lr=2e-5, weight_decay=1e-2):
        self.train_loader = train_loader
        self.val_loader = test_loader  # test_loader is used for validation and evaluation
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load pretrained model and processor
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2).to(self.device)
        # self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-distilled-patch16-224', num_labels=2).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')

        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            print(f"\nEpoch {epoch + 1} starting...")

            # Anaconda prompt output description
            for images, labels, _ in tqdm(self.train_loader, desc = f"Epoch {epoch+1} - Training"):
                inputs = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward Pass
                outputs = self.model(pixel_values=inputs).logits
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

        # self.plot_curves()

    # For evaluation at the end of the training -- NOT NEEDED FOR CALCULATIONS
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validating"):
                inputs = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(pixel_values=inputs).logits
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
    
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    
        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = correct / total
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc * 100:.2f}%")
        return avg_val_loss, val_acc


    def evaluate(self):
        # Load best model
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Evaluating Best Model"):
                inputs = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(pixel_values=inputs).logits
                _, preds = torch.max(outputs, 1)
    
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Classification Report
        report = classification_report(all_labels, all_preds, target_names = ['Real', 'Rendered'], digits = 4)
        print("Classification Report:\n")
        print(report)

        with open("classification_report.txt", "w") as f:
            f.write(report)

if __name__ == "__main__":
    train_loader, test_loader, _, _, _, _ = get_train_test_loaders(
        csv_path='data/train.csv',
        image_folder='data/train_data',
        batch_size=128,
        split_ratio=0.8,
        augmentation=True
    )

    trainer = ViTTrainer(
        train_loader=train_loader,
        test_loader=test_loader,
        save_path='best_vit_model.pth',
        num_epochs=40,
        lr=2e-4,
        weight_decay=0.01
    )

    trainer.train()
    trainer.evaluate()