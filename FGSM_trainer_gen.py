# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 22:36:35 2025

@author: jrdel
"""

import torch
from torchvision import transforms

def fgsm_trainer(model, inputs, labels, device, epsilon=0.05):
    data_denorm = fgsm_trainer_denorm(inputs, device)
    delta = torch.zeros_like(data_denorm, requires_grad=True)
    loss_perturbed = torch.nn.CrossEntropyLoss()(model(data_denorm + delta), labels)
    loss_perturbed.backward()
    perturbed_data = epsilon * delta.grad.detach().sign()
    perturbed_data =  torch.clamp(perturbed_data, 0, 1)
    perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)
    return perturbed_data_normalized
    
    
    
def fgsm_trainer_denorm(batch, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.
        device=pytorch device

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
        
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)