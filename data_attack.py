import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_augmentation import AlbumentationsTransform

def get_attack_transform(image_size=(224, 224), attack_style='SaltAndPepper'):
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
            Posterize: https://explore.albumentations.ai/transform/Posterize
            GaussNoise: https://explore.albumentations.ai/transform/GaussNoise
            RandomShadow: https://explore.albumentations.ai/transform/RandomShadow
      """
      output = None

      if attack_style == 'SaltAndPepper':
        output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),                                                # resize to desired size for consistency
                  A.SaltAndPepper(amount=(0.1, 0.2),                                    # 10-20% of pixels will be noisy
                                  salt_vs_pepper = (0.3, 0.7),
                                  p=1.0),                                               # all images
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # normalize using ImageNet mean and std
                  ToTensorV2()                                                          # convert back to tensor
            ])
        )

      elif attack_style == 'Posterize':
        output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),                                                # resize to desired size for consistency
                  A.Posterize(num_bits=4, p=1.0),                                       # all images
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # normalize using ImageNet mean and std
                  ToTensorV2()                                                          # convert back to tensor
            ])
        )

      elif attack_style == 'GaussNoise':
        output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),                                                # resize to desired size for consistency
                  A.GaussNoise(std_range = (0.01, 0.03),
                                mean_range = (0.0, 0.0),
                                per_channel = True,
                                noise_scale_factor = 1,
                                p=1.0),                                                 # all images
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # normalize using ImageNet mean and std
                  ToTensorV2()                                                          # convert back to tensor
            ])
        )

      elif attack_style == 'RandomShadow':
        output = AlbumentationsTransform(
            A.Compose([
                  A.Resize(*image_size),                                                # resize to desired size for consistency
                  A.RandomShadow(shadow_roi = (0, 0, 1, 1),
                                    num_shadows_limit = (2, 2),
                                    shadow_dimension = 4,
                                    shadow_intensity_range = (0.2, 0.5),
                                    p=1.0),                                             # all images
                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # normalize using ImageNet mean and std
                  ToTensorV2()                                                          # convert back to tensor
            ])
        )

      return output