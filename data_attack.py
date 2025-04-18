import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_augmentation import AlbumentationsTransform

def get_attack_transform(image_size=(224, 224)):
      """
      Transform for test data with adversarial attacks applied.
      Args:
            image_size (tuple): image of size (dim1, dim2)
      Returns:
            output (tensor): AlbumentationsTransform applied to image
      Reference:
            https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/
      Augmentations Used:
            SaltAndPepper: https://explore.albumentations.ai/transform/SaltAndPepper
      """
      output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),              # resize to desired size for consistency
                  A.SaltAndPepper(amount=(0.1, 0.2),  # 10-20% of pixels will be noisy
                                  salt_vs_pepper = (0.3, 0.7),
                                  p=1.0),             # all images
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize using ImageNet mean and std
                  ToTensorV2()                       # convert back to tensor
            ])
      )
      return output