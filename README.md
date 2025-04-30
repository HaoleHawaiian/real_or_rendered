# Real or Rendered
## 1. Project Summary
In recent years, AI-generated deepfake images have become increasingly realistic, making it difficult to differentiate between artificially generated and real, humangenerated images. Current computer vision models have also evolved to discriminate between the two reasonably well, but they are susceptible to adversarial examples that can affect multiple target classes. Bad actors can meaningfully disrupt both classification models and humans, sometimes even with a single attack. This problem is especially relevant in cases such as public trust, political misinformation, cybersecurity, and fraud prevention. We take a cybersecurity approach to discover adversarial attacks that disrupt common pre-trained models, such as EfficientNet and Vision Transformers, and build training methods to effectively handle the attacks. Our objectives are: 1) to find models that can distinguish between real images and AI-Generated ones, 2) to find scalable attack mechanisms that significantly reduce the performance of these models when distinguishing images, and 3) to build and test defense mechanisms to protect against these attacks that also generalize well to other cases. We hypothesize that a targeted white-box attack will be able to efficiently impact classification performance more than a random perturbation in the test data. We also believe that a hybrid defense approach with data augmentation and adversarial training can best help protect against these attacks.


## 2. Approach
Our approach consisted of implementing and testing three image classification models on the image classification problem: EfficientNet, Vision Transformer, and a custom CNN model. Each of these was tested against six configurations: (1) the baseline test dataset, (2) a series of random noise robustness test datasets, (3) an FGSM white-box attack, (4) an FGSM white-box attack with data augmentation, (5) an FGSM white-box attack with adversarial training, and (6) an FGSM white-box attack with hybrid data augmentation and adversarial training.

A key part of our result analysis will include robustness testing, creating adversarial images at the end to try and fool our model. The two perturbations used are [data augmentations](https://explore.albumentations.ai/), and [Fast Gradient Sign Method (FGSM)](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm).

## 4. Datasets
This project was inspired by [this Kaggle competition](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/data), in which authentic images are paired with equivalents generated using generative models. The [Kaggle dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data) can be found here.

## 5. Relevant Files
- models/efficientnetb0.py
- models/juanchitocnn.py
- models/visiontransformer.py
- data_loader.py
- data_attack.py
- data_augmentation.py
- main.ipynb

## 6. Group Members
Juan Raul de la Guardia jguardia7@gatech.edu
Pong-Ravee Halelamien win.halelamien@gatech.edu
Ethan Maluhia Roberts eroberts68@gatech.edu
Anna Zhu azhu95@gatech.edu

## 7. References
[1] Alexander Buslaev, Vladimir I. Iglovikov, Eugene Khvedchenya, Alex Parinov, Mikhail Druzhinin, and Alexandr A. Kalinin. [Albumentations: Fast and flexible image augmentations](https://arxiv.org/pdf/1809.06839). Information, 11(2), 2020  
[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. [An image is worth 16x16 words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929), 2021  
[3] Gamaleldin F. Elsayed, Shreya Shankar, Brian Cheung, Nicolas Papernot, Alex Kurakin, Ian Goodfellow, and Jascha Sohl-Dickstein. [Adversarial Examples that Fool Both Computer Vision and Time-Limited Humans](https://arxiv.org/pdf/1802.08195), 2018  
[4] Ilya Figotin. [Imagenet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000), 2020  
[5] Stanislav Fort. [Multi-attacks: Many images + the same adversarial attack → many target labels](https://arxiv.org/pdf/2308.03792), 2023  
[6] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. [ImageNet-Trained CNNs are Biased Towards Texture; Increasing Shape Bias Improves Accuracy and Robustness](https://arxiv.org/pdf/1811.12231), 2022  
[7] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572), 2015  
[8] Nathan Inkawhich. [Adversarial Example Generation](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html), 2025  
[9] Kaggle. [Detect AI vs. Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/overview). NumPy v1.26 Manual, 2025  
[10] Zico Kolter and Aleksander Madry. Adversarial training, solving the outer minimization, 2025  
[11] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083), 2019  
[12] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. [Do Vision Transformers See Like Convolutional Neural Networks?](https://arxiv.org/pdf/2108.08810), 2022  
[13] Kui Ren, Tianhang Zheng, Zhan Qin, and Xue Liu. [Adversarial Attacks and Defenses in Deep Learning](https://www.sciencedirect.com/science/article/pii/S209580991930503X). Engineering, 6(3):346–360, 2020  
[14] Alessandra Sala, Manuela Jeyaraj, Toma Ijatomi, and Margarita Pitsiani. [Detect AI vs. Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images), 2025  
[15] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108), 2020  
[16] Mingxing Tan and Quoc V. Le. [Efficientnet: Rethinking Model Scaling For Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946), 2020  
[17] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J ´ egou. [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877), 2021  
[18] Florian Tramer, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. [The Space of Transferable Adversarial Examples](https://arxiv.org/pdf/1704.03453), 2017  
[19] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. [Attention is all you need](https://arxiv.org/pdf/1706.03762), 2023  
[20] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A. Efros. [Cnn-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035), 2020  
[21] Bo Yang, Kaiyong Xu, Hengjun Wang, and Hengwei Zhang. [Random Transformation of Image Brightness for Adversarial Attack](https://arxiv.org/pdf/2101.04321), 2021  
[22] Valentina Zantedeschi, Maria-Irina Nicolae, and Ambrish Rawat. [Efficient Defenses Against Adversarial Attacks](https://arxiv.org/pdf/1707.06728), 2017  
