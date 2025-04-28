# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:27:29 2025

@author: jrdel
"""

# import torch
# import torch.nn.functional as F
# from torchvision import transforms

# #fgsm guide taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# def fgsm_attack(image, epsilon, data_grad):
#     '''Args:
#         image: Single image.
#         epsilon: input value representing amount of perturbation
#         data_grad: direction of gradient
        

#     Returns:
#         perturbed_image: image after perturbation is applied'''
#     '''Collect the element-wise sign of the data gradient'''
#     sign_data_grad = data_grad.sign()
#     '''Create the perturbed image by adjusting each pixel of the input image'''
#     perturbed_image = image + epsilon*sign_data_grad
#     '''Adding clipping to maintain [0,1] range'''
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     '''Return the perturbed image'''
#     return perturbed_image

# def denorm(batch, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#     """
#     Convert a batch of tensors to their original scale.

#     Args:
#         batch (torch.Tensor): Batch of normalized tensors.
#         mean (torch.Tensor or list): Mean used for normalization.
#         std (torch.Tensor or list): Standard deviation used for normalization.
#         device=pytorch device

#     Returns:
#         torch.Tensor: batch of tensors without normalization applied to them.
#     """
#     if isinstance(mean, list):
#         mean = torch.tensor(mean).to(device)
#     if isinstance(std, list):
#         std = torch.tensor(std).to(device)

#     return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# def test( model, device, test_loader, epsilon ):

#     # Accuracy counter
#     correct = 0
#     adv_examples = []

#     # Loop over all examples in test set
#     for i, datainput in enumerate(test_loader,0):

#         # Send the data and label to the device
#         data, target = datainput[0].to(device), datainput[1].to(device)

#         # Set requires_grad attribute of tensor. Important for Attack
#         data.requires_grad = True

#         # Forward pass the data through the model
#         output = model(data)
#         init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

#         # If the initial prediction is wrong, don't bother attacking, just move on
#         if init_pred.item() != target.item():
#             continue

#         # Calculate the loss
#         loss = F.cross_entropy(output, target)

#         # Zero all existing gradients
#         model.zero_grad()

#         # Calculate gradients of model in backward pass
#         loss.backward()

#         # Collect ``datagrad``
#         data_grad = data.grad.data

#         # Restore the data to its original scale
#         data_denorm = denorm(data, device)

#         # Call FGSM Attack
#         perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

#         # Reapply normalization
#         perturbed_data_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(perturbed_data)

#         # Re-classify the perturbed image
#         output = model(perturbed_data_normalized)

#         # Check for success
#         final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#         if final_pred.item() == target.item():
#             correct += 1
#             # Special case for saving 0 epsilon examples
#             if epsilon == 0 and len(adv_examples) < 5:
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
#         else:
#             # Save some adv examples for visualization later
#             if len(adv_examples) < 5:
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
#         if i % 200 == 199:    # print every 200 mini-batches
#             print(f'[{i + 1:5d}]')

#     # Calculate final accuracy for this epsilon
#     final_acc = correct/float(len(test_loader))
#     print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

#     # Return the accuracy and an adversarial example
#     return final_acc, adv_examples


import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os

class FGSMAttacker:
    def __init__(self, model, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.model = model
        self.device = device
        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def _denormalize(self, batch):
        return batch * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)

    def _fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return torch.clamp(perturbed_image, 0, 1)
    
    def _show_image_with_confidence(self, image, pred, pred_confidence, save_path=None):
        # Convert tensor to numpy for plotting
        image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
        image = np.clip(image, 0, 1)  # Make sure image values are within valid range
    
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Pred: {pred}, Confidence: {pred_confidence:.2f}%")
        
        if save_path:
            # Make directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()  # Close after saving to avoid memory buildup
        else:
            plt.show()
    
    def attack(self, test_loader, epsilon):
        correct = 0
        adv_examples = []
        shown_images = 0 
    
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
    
            for j in range(data.size(0)):
                image = data[j:j+1].clone().detach().to(self.device)  # [1, C, H, W]
                label = target[j:j+1].to(self.device)
    
                image.requires_grad = True
                output = self.model(image)
                try:
                    init_pred = output.logits.max(1, keepdim=True)[1]
                except:
                    init_pred = output.max(1, keepdim=True)[1]

    
                if init_pred.item() != label.item():
                    continue  
                try:
                    loss = F.cross_entropy(output.logits, label)
                except:
                    loss = F.cross_entropy(output, target)
                
                self.model.zero_grad()
                loss.backward()
    
                data_grad = image.grad.data
                image_denorm = self._denormalize(image)
                perturbed_image = self._fgsm_attack(image_denorm, epsilon, data_grad)
                perturbed_image_norm = self.normalize(perturbed_image)
    
    
                output = self.model(perturbed_image_norm)
                try:
                    final_pred = output.logits.max(1, keepdim=True)[1]
                except:
                    final_pred = output.max(1, keepdim=True)[1]
                
                # Calculate confidence for output image
                softmax_output = F.softmax(output.logits, dim=1)
                confidence = softmax_output[0][final_pred.item()] * 100 
                
                if final_pred.item() == label.item():
                    correct += 1
                    if epsilon == 0 and len(adv_examples) < 5:
                        adv_examples.append((init_pred.item(), final_pred.item(),
                                             perturbed_image.squeeze().detach().cpu().numpy()))
                    # Only show 3 images
                    if shown_images < 3 and confidence < 15:   
                        save_path = f"adv_examples/epsilon_{epsilon}_img_{shown_images+1}.png"
                        self._show_image_with_confidence(perturbed_image, final_pred.item(), confidence, save_path=save_path)
                        shown_images += 1
                else:
                    if len(adv_examples) < 5:
                        adv_examples.append((init_pred.item(), final_pred.item(),
                                             perturbed_image.squeeze().detach().cpu().numpy()))
                    if shown_images < 3 and confidence < 15:   #
                        save_path = f"adv_examples/epsilon_{epsilon}_img_{shown_images+1}.png"
                        self._show_image_with_confidence(perturbed_image, final_pred.item(), confidence, save_path=save_path)
                        shown_images += 1
    
            if i % 2000 == 1999:
                print(f'[{i + 1:5d}]')
    
        final_acc = correct / float(len(test_loader.dataset))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")
        return final_acc, adv_examples