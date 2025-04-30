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
[13] Kui Ren, Tianhang Zheng, Zhan Qin, and Xue Liu. [Adversarial Attacks and Defenses in Deep Learning](https://pdf.sciencedirectassets.com/314095/1-s2.0-S2095809920X0004X/1-s2.0-S209580991930503X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBYaCXVzLWVhc3QtMSJIMEYCIQCdfcJFlq2xRJnirTGDZzQE1mozCOQgHA%2FiNht3haYvdgIhAKvkI1wZUKNXFES69qUJT%2FdwVvRpGdH0fUQhqnyEVzhrKrsFCK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwInVufUJHTmDXtOqoqjwUyRdfV9%2F4b%2FI3AfCX6ODjxgCgoK3SraO%2Ffsxy%2BPN2L13tvLopL3w6Ofb%2FnzVgAB30t1Fphs%2FYHaXVo94t4V33eh0w7FTLePF4Pa%2FQjSZp6ApoPoWMi2YFIQ9dlTOSMc0mXeLLL8BeNsq1rIehoIVvC27JGvCwRi6eGUx0MwfqXKC%2BPhUELk3d1ArIcwAoFtIIvGbV0ZotkZcbxcV4Oc7lw83%2FToTA4A7XoY2iYaObFWaqM0%2FkKV%2F2Dbs5mZU%2F0tI3droGqvZcxotNZC96Eun38aFVk0yGTmsCvose%2FvBaqVw8L4XXAUXKqy1rNd9L1L1GsxfyQRQwoI8ZC%2FkOA1l3zKc9ekPSXwuYV8srzRSRBw5ULQMTDXwosXTkX2lumK2jZanBYRATh16x9fSIQkjY%2FIImjTqR8Wb1HkyNnWUAwRAC9BSOnbvSQ0ANOVtysYLVTGCxCGPJszASDKnbotTTnFSxxS8gjK%2BYa0P0tkOjXel%2F7tMNGMRK75DBPmYMeWr%2FBPr7znBojycmMB99QG9Xc6aWqrGVlGNmEJpIuhLdJJ0gxfDzPcCFrhSJyxpXiuZdY7xTbqUIlZsHaPh7urXf%2F3KI7cvRe8Be2a5Sno6enE82L5tA2w8EYpAmSWT4vlYxMQ%2B2ASxIpWl54OKvD%2Bg%2Bdg602G%2BpakuARUFw4ALjSytzF5kgukVAwVMgJSiCm5P9%2Bg3htpg9dHNalmj2jwhlarA8QJxkUXxqlqa8izC%2F8%2BvHcY0cL45rRLrJOKbi9Dh5k7ZoOXjr40MydbqtVxN7c5%2F7kUXdmN%2FjZj2XPeDOOD9DrhREKkrBb4jmUouIIm7%2FnXY6J2oKl5D9ZX71NMRzU85QPuXKcJwGYvOkttzd1MOyhysAGOrABiYaMNqh6%2BmmZwe1urs%2FvPSd6zYF%2BIh2OyrleThLta%2BzSU0K5vDAYhPlx83zRg%2FtXYRQ2%2BwCLcKqb3kbjkO6Ck9k6%2B4IL3WJsBFzZ%2FyuOl0kjVBW%2B1OmbmnqDtnTnwWk%2FkdbDJN0hwggmacSaq8Lckf3uwuOCU9GSvR%2BufQX71hfsB58Xpsq4Q9wZkshCXmh46a5YPZPPNF4RW5Edy52G7K9n6m7EbfgA8vAc8bXJVrs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250430T220716Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2BDU7MGS%2F20250430%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=455f25b91b0da52be5819e98feb9f45e013bcadff89c919273a3c2eab90c3392&hash=013f9fbcd8fcbcaaf520553a2461c0d5d32f228a7eec2b8d08e69948530fd99a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S209580991930503X&tid=spdf-c5aa122f-b4b5-4157-b292-d9493ab5455c&sid=7ca948a591b338498f4b4b045e626c9f984agxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f1559575902525b065601&rr=938a59e1fcea7ebf&cc=us). Engineering, 6(3):346–360, 2020  
[14] Alessandra Sala, Manuela Jeyaraj, Toma Ijatomi, and Margarita Pitsiani. [Detect AI vs. Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images), 2025  
[15] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108), 2020  
[16] Mingxing Tan and Quoc V. Le. [Efficientnet: Rethinking Model Scaling For Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946), 2020  
[17] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J ´ egou. [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877), 2021  
[18] Florian Tramer, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. [The Space of Transferable Adversarial Examples](https://arxiv.org/pdf/1704.03453), 2017  
[19] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. [Attention is all you need](https://arxiv.org/pdf/1706.03762), 2023  
[20] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A. Efros. [Cnn-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035), 2020  
[21] Bo Yang, Kaiyong Xu, Hengjun Wang, and Hengwei Zhang. [Random Transformation of Image Brightness for Adversarial Attack](https://arxiv.org/pdf/2101.04321), 2021  
[22] Valentina Zantedeschi, Maria-Irina Nicolae, and Ambrish Rawat. [Efficient Defenses Against Adversarial Attacks](https://arxiv.org/pdf/1707.06728), 2017  
