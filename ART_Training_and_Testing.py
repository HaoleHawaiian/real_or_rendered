# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:03:05 2025

@author: jrdel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

import os
import pandas as pd
import numpy as np
from PIL import Image


class ARTProcess():
    def __init__(self, project, epochs=5, learning_rate=0.001, batch_size=16, optimizer='SGD', momentum=0.9, weight_decay=0.01, size=80000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = 2
        self.project = project
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.project.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.project.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.size=size
        self.train_arr=None
        self.train_labels=None
        self.test_arr=None
        self.test_labels=None
        self.classifier=None
        
        self.attack=None
        self.x_test_adv=None
        self.defense=None



    def load_data(self, csv_path='data/train.csv', image_folder='data/train_data/'):
        train_labels=pd.read_csv(csv_path)
    
        filename_list=[]
        image_list=[]
        width=[]
        height=[]
    
        for filename in os.listdir(image_folder):
            img=Image.open(os.path.join(image_folder, filename))
            width.append(img.size[0])
            height.append(img.size[1])
            if len(width) % 100 == 0:
                print(len(width))
            resized_image = img.resize((224, 224))
            img_arr=np.array(resized_image,dtype='int32')
            if img_arr.shape==(224,224):
                img_arr=np.expand_dims(img_arr,2)
                img_arr=np.pad(img_arr,pad_width=((0,0),(0,0),(0,2)),mode='constant',constant_values=0)
            filename_list.append('train_data/'+filename)
            image_list.append(img_arr)
            img.close()
            if len(image_list)==self.size:
                break
        image_array=np.array(image_list)
    
        image_df=pd.DataFrame({'file_name':filename_list, 'image':image_list})
    
        image_df_labeled=image_df.merge(train_labels, on='file_name')
    
        label_array=np.array(image_df_labeled['label'])
    
        # print(f"Format: {train_image_1.format}")
        # print(f"Size: {train_image_1.size}")
        # print(f"Mode: {train_image_1.mode}")
    
        # print(label_array)
    
        encoded_arr = np.zeros((label_array.size, label_array.max()+1), dtype=int)
        encoded_arr[np.arange(label_array.size),label_array] = 1 
        
        train_size=int(len(image_list)*.8)
        train_arr=image_array[0:train_size]
        test_arr=image_array[train_size:]
        train_labels=encoded_arr[0:train_size]
        test_labels=encoded_arr[train_size:]
        train_arr = np.transpose(train_arr, (0, 3, 1, 2)).astype(np.float32)
        test_arr = np.transpose(test_arr, (0, 3, 1, 2)).astype(np.float32)
        
        self.train_arr=train_arr
        self.test_arr=test_arr
        self.train_labels=train_labels
        self.test_labels=test_labels



    def create_classifier(self):
        classifier = PyTorchClassifier(
            model=self.project.model.to(self.device),
            #clip_values=(min_pixel_value, max_pixel_value),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(3, 224, 224),
            nb_classes=2,
        )
        self.classifier=classifier



    def train_classifier(self):
        self.classifier.fit(self.train_arr, self.train_labels, batch_size=self.batch_size, nb_epochs=self.epochs)
        print('fit done')

    def predict_classifier(self):
        predictions = self.classifier.predict(self.test_arr)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.test_labels, axis=1)) / len(self.test_labels)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))



    def generate_adversarials(self):
        self.attack = FastGradientMethod(estimator=self.classifier, batch_size=self.batch_size, eps=0.2)
        self.x_test_adv = self.attack.generate(x=self.test_arr)
        
    
    
    def evaluate_adversarials(self):
        predictions = self.classifier.predict(self.x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.test_labels, axis=1)) / len(self.test_labels)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        
    
    def build_defense(self):
        self.defense = AdversarialTrainer(classifier=self.classifier, attacks=self.attack, ratio=1)
        
    
    # Step 9: Train the defense
    def train_defense(self):
        self.defense.fit(self.train_arr, self.train_labels, batch_size=self.batch_size, nb_epochs=self.epochs)
    
    # Step 10: Evaluate defense classifier on adversarial examples
    def evaluate_adversarial_training(self):
        predictions = self.defense.predict(self.x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.test_labels, axis=1)) / len(self.test_labels)
        print("Adversarial Training accuracy on adversarial test examples: {}%".format(accuracy * 100))















