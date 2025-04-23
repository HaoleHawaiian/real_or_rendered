import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Takes raw images from 'data/' folder, applies data augmentation using albumentations,
and creates train/test datasets.
"""

class AlbumentationsTransform:
      """
      Albumentations returns numpy arrays, so we need to convert them to PyTorch tensors

      Args:
            img: numpy array
      Returns:
            int: fkdsjfdsf
      """
      def __init__(self, augmentations):
            self.augment = augmentations                                                  # augmentation from albumentations

      def __call__(self, img):
            """
            Convert image (PIL) to numpy array, then applies augmentation
            """
            img = np.array(img)
            augmented = self.augment(image=img)                                           # augmented is a dictionary with keys like 'image'
            return augmented['image']

def get_train_transform(image_size=(224, 224)):
      """
      Augmentation for training set
      Args:
            image_size (tuple): image of size (dim1, dim2)
      Returns:
            output (tensor): AlbumentationsTransform applied to image
      Reference:
            https://explore.albumentations.ai/
            Check out the different transformations here. Feel free to change them.
      """
      output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),                                                  # resize to desired size for consistency
                  A.HorizontalFlip(p=0.5),                                                # flip horizontally with p% chance
                  A.RandomBrightnessContrast(p=0.2),                                      # bright and contrast with p% chance
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     # normalize using ImageNet mean and std
                  ToTensorV2()                       # convert back to tensor
            ])
      )
      return output

def get_test_transform(image_size=(224, 224)):
      """
      Transform for test data. No augmentation applied
      Args:
            image_size (tuple): image of size (dim1, dim2)
      Returns:
            output (tensor): AlbumentationsTransform applied to image
      """
      output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),                                                  # resize to desired size for consistency
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     # normalize using ImageNet mean and std
                  ToTensorV2()                                                            # convert back to tensor
            ])
      )
      return output