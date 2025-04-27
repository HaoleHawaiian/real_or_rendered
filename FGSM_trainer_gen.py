# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 22:36:35 2025

@author: jrdel
"""

import torch
from torchvision import transforms

def fgsm_trainer(model, inputs, labels, device, epsilon=0.1, normalization=True):
    if normalization:
        data = fgsm_trainer_denorm(inputs, device)
    else:
        data=inputs
    delta = torch.zeros_like(data, requires_grad=True)
    loss_perturbed = torch.nn.CrossEntropyLoss()(model(data + delta), labels)
    loss_perturbed.backward()
    fgsm_result = epsilon * delta.grad.detach().sign()
    perturbed_image = (inputs + fgsm_result)
    if normalization:
        perturbed_image_result = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_image)
    else:
        perturbed_image_result=perturbed_image
    return perturbed_image_result

def fgsm_trainer_iter(model, inputs, labels, device, epsilon=0.1, alpha=0.01, iterations=20, normalization=True):
    if normalization:
        data = fgsm_trainer_denorm(inputs, device)
    else:
        data=inputs
    delta = torch.zeros_like(data, requires_grad=True)
    for i in range(iterations):
    
        loss_perturbed = torch.nn.CrossEntropyLoss()(model(data + delta), labels)
        loss_perturbed.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    fgsm_result=delta.detach()
    perturbed_image = (inputs + fgsm_result)
    if normalization:
        perturbed_image_result = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_image)
    else:
        perturbed_image_result=perturbed_image
    return perturbed_image_result
    
    
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