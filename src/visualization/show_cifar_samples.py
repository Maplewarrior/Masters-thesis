import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch

def load_cifar10_samples():
    """Load CIFAR10 dataset and return samples"""
    # Simple transform to convert PIL to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download and load CIFAR10 dataset
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Get class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return dataset, classes

def show_samples_grid(dataset, classes, num_samples=30):
    """Display samples in a 3x10 grid"""
    # Create figure with subplots
    fig, axes = plt.subplots(3, 10, figsize=(15, 6))
    fig.suptitle('CIFAR10 Dataset Samples', fontsize=16)
    
    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    
    for i, idx in enumerate(indices):
        row = i // 10
        col = i % 10
        
        # Get image and label
        image, label = dataset[idx]
        
        # Convert tensor to numpy and transpose for matplotlib (CHW -> HWC)
        image_np = image.permute(1, 2, 0).numpy()
        
        # Display image
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f'{classes[label]}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, axes

if __name__ == '__main__':
    print("Loading CIFAR10 dataset...")
    dataset, classes = load_cifar10_samples()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    print("Displaying 30 random samples in 3x10 grid...")
    fig, axes = show_samples_grid(dataset, classes) 
    plt.savefig('cifar10_samples.pdf', dpi=200)