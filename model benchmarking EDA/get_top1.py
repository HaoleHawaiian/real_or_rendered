import torch
import models.juanchitocnn
import data_loader
"""
Get the Top-1 accuracy of a model for reporting
"""

def compute_accuracy(model, train_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in train_loader:
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

regular_train_loader, regular_test_loader = data_loader.data_to_train_test_dataloaders(csv_path='data/train.csv', 
                                                                                       image_folder='data/train_data', 
                                                                                       image_size=(224, 224), 
                                                                                       split_ratio=train_test_split_ratio, 
                                                                                       train_batch_size=batch_size, 
                                                                                       test_batch_size=batch_size)

project_model.data_load(regular_train_loader, regular_test_loader)
# regular_correct, total, regular_incorrect_preds = project_model.evaluate()

top1_accuracy = compute_accuracy(project_model, regular_train_loader)
print(f"The Top-1 accuracy for {select_model} is {top1_accuracy}.")