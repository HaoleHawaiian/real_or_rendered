# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:25:38 2025

@author: aloha
"""

import torch
from vit import ViTTrainer
from FGSM_attack import FGSMAttacker
from data_loader import get_train_test_loaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (same way as you trained)
    _, test_loader, _, _, _, _ = get_train_test_loaders(
        csv_path='data/train.csv',
        image_folder='data/train_data',
        # batch_size=1,  # FGSM processes one image at a time
        split_ratio=0.8,
        augmentation=False
    )

    # Load the best model from running vit.py by itself
    trainer = ViTTrainer(None, test_loader, save_path='best_vit_model.pth')  
    trainer.model.load_state_dict(torch.load(trainer.save_path, map_location=device))
    trainer.model.to(device)
    trainer.model.eval()

    # Run FGSM Attack
    attacker = FGSMAttacker(trainer.model, device)
    epsilon = 0.1
    acc, adv_examples = attacker.attack(test_loader, epsilon)

    print(f"FGSM Attack completed. Accuracy after attack: {acc*100:.2f}%")

if __name__ == "__main__":
    main()