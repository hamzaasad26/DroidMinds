import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2

# Define paths
base_dir = 'content/Diabetic_Balanced_Data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32
max_images_per_class = 100  # Limit to 100 images per class

# Function to load a limited number of images from each class folder
def load_limited_images(directory, max_images_per_class):
    data = []
    labels = []
    class_folders = sorted(os.listdir(directory))
    
    for class_folder in class_folders:
        class_path = os.path.join(directory, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            for img_name in images[:max_images_per_class]:  # Limit to max_images_per_class
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_height, img_width))
                img = img / 255.0  # Normalize
                data.append(img)
                labels.append(int(class_folder))
    
    return np.array(data), np.array(labels)

# Load limited images for training and validation
X_train, y_train = load_limited_images(train_dir, max_images_per_class)
X_val, y_val = load_limited_images(val_dir, max_images_per_class)
X_test, y_test = load_limited_images(test_dir, max_images_per_class)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained EfficientNet-B3 model
model = models.efficientnet_b3(pretrained=True)

# Modify the classifier to match the number of classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    
    return model

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

# Fine-tuning: Unfreeze some layers
for param in model.parameters():
    param.requires_grad = False
for param in model.features[-10:].parameters():
    param.requires_grad = True

# Re-define optimizer with a lower learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Fine-tune the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
test_corrects = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)

test_loss /= len(test_loader.dataset)
test_acc = test_corrects.double() / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Save the model
torch.save(model.state_dict(), 'efficientnet_b3_model.pth')
print("Model saved as efficientnet_b3_model.pth")