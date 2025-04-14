# Real or Rendered
## 1. Project Summary
Several years ago, AI images were easy to spot, as they were generally nonsensical or overly cartoonish. Now, with recent AI technological advancements, AI generated imagery is increasingly difficult to differentiate from real images, or images created by a human. This is especially relevant in cases of public opinion and political misinformation, as well as security and fraud prevention. We seek to use deep learning methods to help distinguish between the images to establish authenticity and transparency.

This project was inspired by [this Kaggle competition](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/data), in which authentic images are paired with equivalents generated using generative models. 

## 2. Approach
To accomplish our goal of real vs rendered image classification, we plan on using several well established convolutional neural network architectures, generative adversarial networks, custom architectures, and vision transformer hybrids to make these determinations.

A key part of our result analysis will include robustness testing, creating adversarial images at the end to try and fool our model. Examples of perturbations include [Fast Gradient Sign Method (FGSM)](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm), Projected Gradient Descent (PGD), [DeepFool](https://medium.com/machine-intelligence-and-deep-learning-lab/a-review-of-deepfool-a-simple-and-accurate-method-to-fool-deep-neural-networks-b016fba9e48e), [Carlini-Wagner attacks](https://medium.com/@zachariaharungeorge/adversarial-attacks-with-carlini-wagner-approach-8307daa9a503), and [white, gray, and black box attacks](https://posts.specterops.io/learning-machine-learning-part-2-attacking-white-box-models-1a10bbb4a2ae).


## 3. Resources and Related Work
The most famous example of a neural network architecture that generates its own images to attempt to fool a network is the [Generative Adversarial Network](https://arxiv.org/pdf/1701.00160). This network trains a generator to make fake images, and a discriminator to distinguish them.

Among specific models used to distinguish real or fake images, we can point to the research-based [DIRE](https://arxiv.org/pdf/2303.09295) or [De-FAKE](https://arxiv.org/pdf/2210.06998) detectors, as well as the commercial detectors [Hive AI](https://thehive.ai/), [Optic](https://www.aiornot.com/), and [Illuminarty](https://illuminarty.ai/en/), which were compared by [Ha et. al](https://arxiv.org/pdf/2402.03214). Other methodologies include the uncertainty based [WePe](https://arxiv.org/pdf/2412.05897v1), and the perturbation based [RIGID](https://arxiv.org/pdf/2405.20112).

## 4. Datasets
[Kaggle dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data)

## 5. Group Members
Juan Raul de la Guardia jguardia7@gatech.edu
Pong-Ravee Halelamien win.halelamien@gatech.edu
Ethan Maluhia Roberts eroberts68@gatech.edu
Anna Zhu azhu95@gatech.edu
