import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt

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
        return torch.clamp(perturbed_image, 0, 1), epsilon * sign_data_grad
    
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
        original_images=[]
        adv_examples = []
        perturbation_list = []
        perturbation_list_2 = []
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
                perturbed_image, perturbation = self._fgsm_attack(image_denorm, epsilon, data_grad)
                perturbed_image_norm = self.normalize(perturbed_image)
    
    
                output = self.model(perturbed_image_norm)
                try:
                    final_pred = output.logits.max(1, keepdim=True)[1]
                except:
                    final_pred = output.max(1, keepdim=True)[1]
                
                # Calculate confidence for output image
                try:
                    softmax_output = F.softmax(output.logits, dim=1)
                except:
                    softmax_output = F.softmax(output, dim=1)
                
                confidence = softmax_output[0][final_pred.item()] * 100 
                confidence_threshold = 40
                
                if final_pred.item() == label.item():
                    correct += 1
                    if epsilon == 0 and len(adv_examples) < 5:
                        adv_examples.append((init_pred.item(), final_pred.item(),
                                             perturbed_image.squeeze().detach().cpu().numpy()))
                else:
                    if len(adv_examples) < 5:
                        original_images.append((init_pred.item(), final_pred.item(),
                                             image_denorm.squeeze().detach().cpu().numpy()))
                        adv_examples.append((init_pred.item(), final_pred.item(),
                                             perturbed_image.squeeze().detach().cpu().numpy()))

                # if confidence < confidence_threshold and shown_images < 5:
                #     save_path = f"adv_examples/epsilon_{epsilon}_img_{shown_images+1}.png"
                #     self._show_image_with_confidence(perturbed_image, final_pred.item(), confidence, save_path=save_path)
                #     shown_images += 1
                    
                # if shown_images < 3 and confidence < 40:
                #     save_path = f"adv_examples/epsilon_{epsilon}_img_{shown_images+1}.png"
                #     self._show_image_with_confidence(perturbed_image, final_pred.item(), confidence, save_path=save_path)
                #     shown_images += 1

                    perturbation_list.append((init_pred.item(), final_pred.item(),
                                             perturbation.squeeze().detach().cpu().numpy()))
                    perturbation_list_2.append((init_pred.item(), final_pred.item(),
                                             (perturbed_image-image_denorm).squeeze().detach().cpu().numpy()))
                    # if shown_images < 3 and confidence < 15:   #
                    #     save_path = f"adv_examples/epsilon_{epsilon}_img_{shown_images+1}.png"
                    #     self._show_image_with_confidence(perturbed_image, final_pred.item(), confidence, save_path=save_path)
                    #     shown_images += 1

    
            if i % 2000 == 1999:
                print(f'[{i + 1:5d}]')
    
        final_acc = correct / float(len(test_loader.dataset))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")
        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(adv_examples)):
            _,_,origimg = original_images[i]
            _,_,perturb = perturbation_list[i]
            orig,adv,ex = adv_examples[i]
            image_array=[origimg,perturb*5,ex]
            for j in range(3):
                cnt += 1
                plt.subplot(len(adv_examples),3,cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j==0:
                    plt.ylabel(f"Image:{i}" , fontsize=14)
                    plt.title(f"Original Image")
                if j==1:
                    plt.title(f"Applied Perturbation, Scaled x5")
                if j==2:
                    plt.title(f"Perturbed Image")
                plt.imshow(np.moveaxis(image_array[j], 0, -1))
        plt.tight_layout()
        plt.show()
        return final_acc, original_images