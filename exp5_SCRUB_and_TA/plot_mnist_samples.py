# This code just plots some MNIST samples for the paper.
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the Python path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp5_SCRUB_and_TA.prepare_image_data_v2 import download_dataset

def plot_mnist_samples():
    """
    Downloads the MNIST dataset and plots three sample images from each class.
    """
    # Create a directory to save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Download MNIST dataset
    print("Downloading MNIST dataset...")
    train_dataset, _ = download_dataset(root_dir='data', dataset_name='MNIST')
    print("Dataset downloaded.")

    # Find three sample images for each class
    print("Finding samples for each class...")
    class_samples = {i: [] for i in range(10)}
    # The dataset is a ConcatDataset, we need to access the underlying dataset
    dataset = train_dataset.datasets[0] if hasattr(train_dataset, 'datasets') else train_dataset

    for img, label in dataset:
        if isinstance(label, int) and len(class_samples[label]) < 3:
            class_samples[label].append(img)
        if all(len(v) == 3 for v in class_samples.values()):
            break
    print("Samples found.")

    # Create the plot
    print("Generating plot...")
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.5))
    fig.suptitle('MNIST Dataset Samples', fontsize=16)

    for col in range(10):
        axes[0, col].set_title(f"Class {col}", fontsize=10)
        for row in range(3):
            ax = axes[row, col]
            if col < len(class_samples) and row < len(class_samples[col]):
                image = class_samples[col][row]
                ax.imshow(image, cmap='gray')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, "mnist_samples.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_mnist_samples() 