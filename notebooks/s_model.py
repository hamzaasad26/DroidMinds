# MODEL = 45% sample, 5 epochs, 5 fine-tuning epochs, efficientnet_b3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

# Define paths
base_dir = 'content/Diabetic_Balanced_Data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def load_datasets(sample_ratio=1.0):
    """
    Load datasets with option to use a smaller subset for faster training
    """
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])
    
    # Optionally use a subset of the data for faster training
    if sample_ratio < 1.0:
        print(f"Using {sample_ratio*100:.1f}% of training data for faster execution")
        train_size = int(len(train_dataset) * sample_ratio)
        indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = Subset(train_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Print dataset sizes and class info
    # For subset, we need to get classes from the original dataset
    if isinstance(train_dataset, Subset):
        classes = train_dataset.dataset.classes
        original_length = len(train_dataset.dataset)
        subset_length = len(train_dataset)
        print(f"Original training set size: {original_length} images")
        print(f"Using subset of: {subset_length} images ({subset_length/original_length:.1%})")
    else:
        classes = train_dataset.classes
        print(f"Training set size: {len(train_dataset)} images")
    
    print("\n--- Dataset Information ---")
    print(f"Validation set size: {len(val_dataset)} images")
    print(f"Test set size: {len(test_dataset)} images")
    print(f"Classes: {classes}")
    print("---------------------------\n")
    
    # Print class distribution
    if not isinstance(train_dataset, Subset):
        class_counts = {cls: 0 for cls in classes}
        for _, label in train_dataset.samples:
            class_counts[classes[label]] += 1
        print("Class distribution in training set:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count} images")
        print()
    
    return train_loader, val_loader, test_loader, classes

# Function to plot training history
def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

# Training function with detailed progress tracking and checkpoint saving
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, phase="training", save_path=None,
                checkpoint_frequency=1, resume_from=None):
    best_val_acc = 0.0
    start_epoch = 0
    
    # Initialize or load history
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        history = checkpoint.get('history', {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        })
        print(f"Resuming from epoch {start_epoch} with validation accuracy {checkpoint.get('val_acc', 0):.4f}")
    else:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\n[{phase}] Epoch {epoch+1}/{start_epoch + num_epochs}")
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Training")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            batch_loss = loss.item() * inputs.size(0)
            batch_corrects = torch.sum(preds == labels.data).double()
            
            running_loss += batch_loss
            running_corrects += batch_corrects
            
            # Update progress bar with current batch loss and accuracy
            batch_acc = batch_corrects / inputs.size(0)
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc:.4f}"
            })
        
        dataset_size = len(train_loader.dataset) if not isinstance(train_loader.dataset, Subset) else len(train_loader.dataset.indices)
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size
        
        # Store training metrics
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        # Progress bar for validation
        val_bar = tqdm(val_loader, desc=f"Validation")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data).double()
                
                val_loss += batch_loss
                val_corrects += batch_corrects
                
                # Update progress bar
                batch_acc = batch_corrects / inputs.size(0)
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{batch_acc:.4f}"
                })
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        
        # Store validation metrics
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\n[{phase}] Epoch {epoch+1}/{start_epoch + num_epochs} Summary:")
        print(f"  Training   - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % checkpoint_frequency == 0 and save_path:
            checkpoint_path = f"{save_path.split('.')[0]}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'history': history
            }, checkpoint_path)
            print(f"  Saved checkpoint to: {checkpoint_path}")
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'history': history
            }, save_path)
            print(f"  Saved best model with validation accuracy: {val_acc:.4f}")
    
    total_time = time.time() - total_start_time
    print(f"\n[{phase}] Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plot_path = f"{save_path.split('.')[0]}_history.png" if save_path else None
    plot_training_history(history, save_path=plot_path)
    
    return model, history

def evaluate_model(model, test_loader, criterion, classes):
    print("\n=== FINAL MODEL EVALUATION ===")
    
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
    test_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print("\nTest Set Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    
    # Class-wise accuracy
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def main():
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Parse command-line arguments (or set default values)
    import argparse
    parser = argparse.ArgumentParser(description='Train a model for diabetic retinopathy classification')
    parser.add_argument('--model', type=str, default='mobilenet_v2', 
                        choices=['efficientnet_b0', 'efficientnet_b3', 'mobilenet_v2', 'resnet18'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for initial training')
    parser.add_argument('--ft_epochs', type=int, default=5, help='Number of epochs for fine-tuning')
    parser.add_argument('--sample_ratio', type=float, default=0.45, 
                        help='Fraction of training data to use (0-1)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint file')
    parser.add_argument('--evaluate_only', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--checkpoint_freq', type=int, default=1, 
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Load datasets with optional sampling for faster execution
    train_loader, val_loader, test_loader, classes = load_datasets(sample_ratio=args.sample_ratio)
    
    # Check if CUDA is available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize selected model
    print(f"\nInitializing {args.model} model...")
    
    if args.model == 'efficientnet_b3':
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        if args.model == 'efficientnet_b3':
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    elif args.model == 'efficientnet_b0':
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    elif args.model == 'mobilenet_v2':
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    elif args.model == 'resnet18':
        model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes))
    
    model = model.to(device)
    print(f"Model output layer modified for {len(classes)} classes")
    
    # Define file paths
    model_base_name = f"{args.model}_dr_classifier"
    initial_model_path = f"{model_base_name}_initial.pth"
    final_model_path = f"{model_base_name}_finetuned.pth"
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Skip training if evaluate_only flag is set
    if args.evaluate_only and os.path.exists(final_model_path):
        print(f"Loading model from {final_model_path} for evaluation only...")
        checkpoint = torch.load(final_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # PHASE 1: Initial Training (or resume from checkpoint)
        print("\n=== STARTING INITIAL TRAINING ===")
        model, initial_history = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=args.epochs, 
            phase="Initial Training", 
            save_path=initial_model_path,
            checkpoint_frequency=args.checkpoint_freq,
            resume_from=args.resume
        )
        
        print(f"\nInitial training completed. Best model saved to {initial_model_path}")
        
        # PHASE 2: Fine-tuning
        print("\n=== STARTING FINE-TUNING ===")
        print("Loading the best initial model...")
        
        # Load the saved model
        checkpoint = torch.load(initial_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.4f}")
        
        # Freeze most layers and only train the last few layers
        print("Freezing early layers...")
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last few layers for fine-tuning
        # The structure differs between model architectures
        if args.model.startswith('efficientnet'):
            print("Unfreezing the last few feature layers and classifier...")
            for param in model.features[-5:].parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.model == 'mobilenet_v2':
            print("Unfreezing the last few feature layers and classifier...")
            for param in model.features[-3:].parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.model == 'resnet18':
            print("Unfreezing the last layer and classifier...")
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        
        # Re-define optimizer with a lower learning rate for fine-tuning
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        print("Optimizer reset with learning rate 0.0001")
        
        # Fine-tune the model
        model, finetuning_history = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=args.ft_epochs, 
            phase="Fine-tuning", 
            save_path=final_model_path,
            checkpoint_frequency=args.checkpoint_freq
        )
        
        print(f"\nFine-tuning completed. Best model saved to {final_model_path}")
    
    # Load the best fine-tuned model for evaluation
    if os.path.exists(final_model_path):
        checkpoint = torch.load(final_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate the model
    evaluate_model(model, test_loader, criterion, classes)
    
    print("Training and evaluation completed!")
    if not args.evaluate_only:
        print(f"Initial model saved to: {initial_model_path}")
        print(f"Fine-tuned model saved to: {final_model_path}")
    print(f"Training plots and confusion matrix saved to current directory")


if __name__ == "__main__":
    main()