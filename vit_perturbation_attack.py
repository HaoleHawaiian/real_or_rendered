# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:12:22 2025

@author: aloha
"""

from vit import ViTTrainer
from data_loader import get_train_test_loaders
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_style = 'RandomShadow'

    # Load the attack loader with SaltAndPepper noise applied
    _, _, attack_loader, _, _, _ = get_train_test_loaders(
        csv_path = 'data/train.csv',
        image_folder = 'data/train_data',
        split_ratio = 0.8,
        augmentation = False,
        attack_style = attack_style
    )

    # Load the trained ViT model
    trainer = ViTTrainer(None, attack_loader, save_path='best_vit_model.pth')  
    trainer.model.load_state_dict(torch.load(trainer.save_path, map_location=device))
    trainer.model.to(device)
    trainer.model.eval()

    # Run evaluation on perturbed data
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels, _ in attack_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trainer.model(images)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Accuracy on {attack_style} perturbed data: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
