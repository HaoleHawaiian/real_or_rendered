import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import models.juanchitocnn
import data_loader



"""
Get the Top-1 accuracy of a model for reporting.
Use ImageNet-1k validation set.
You must download validation set separately https://image-net.org/download.php
Add the validation set to the gitignore so it's not uploaded to github.
"""

def compute_accuracy(model, loader, device='cuda'):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

select_model = 'JuanchitoCNN' # ['EfficientNet', 'VisionTransformer', 'JuanchitoCNN']
train_original = True # Re-train model?
train_augmentation = False

# Model Hyperparameters
epochs = 5
learning_rate = 0.001
batch_size = 32
weight_decay = 0.01
momentum = 0.9
optimizer = 'SGD' # ['SGD', 'AdamW']

# Other parameters
train_test_split_ratio = 0.8

# Run
if select_model == 'JuanchitoCNN':
    project_model = models.juanchitocnn.ProjectJuanchitoCNN(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, optimizer=optimizer, momentum=momentum, weight_decay=weight_decay)

# (assuming you already defined batch_size somewhere)
_, validation_loader = data_loader.data_to_train_test_dataloaders(
    csv_path='../data/train.csv',
    image_folder='../data/train_data',
    # batch_size=batch_size
)

project_model.model.load_state_dict(torch.load('../JuanchitoCNN.pth'))
project_model.model.eval()

# Now pass both to compute_accuracy
top1_accuracy = compute_accuracy(project_model, validation_loader)

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")


