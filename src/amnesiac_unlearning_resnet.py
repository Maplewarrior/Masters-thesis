
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from scipy import ndimage
from PIL import Image
import copy
import random
import time
import pickle
import os

torch.set_printoptions(precision=3)

def get_device():
    """Get the device to use for training (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"Using device: {device}")

def normalize(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    trans = np.transpose(npimg, (1,2,0))
    return np.squeeze(trans)


def imshow(img):
    temp = normalize(img)
    plt.imshow(temp, vmin=0, vmax=1, cmap='Greys_r')
    plt.show()


# ================================================
# Data entry and processing
# ================================================

# Transform image to tensor and normalize features from [0,255] to [0,1]
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,),(0.5,)),
                                ])

# Define a writable directory
data_dir = "./data"


# Patch the MNIST dataset to return indices. This is needed for the Amnesiac model to work.
# The amnesiac model tracks the indices of the data that is used for training, so we need to return them.
class MNIST(datasets.MNIST):
    def __init__(self, *args, use_indices=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_indices = use_indices
        if self.use_indices:
            self.indices = list(range(len(self.data)))
    
    def __getitem__(self, idx):
        original_output = super().__getitem__(idx)
        if self.use_indices:
            idxs = self.indices[idx]
            return original_output + (idxs,)
        return original_output

# Download MNIST into the writable directory
traindata = MNIST(root=data_dir, train=True, download=True, transform=transform, use_indices=True)
testdata = MNIST(root=data_dir, train=False, download=True, transform=transform, use_indices=True)


# Loaders that give 64 example batches
all_data_train_loader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)
all_data_test_loader = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=True)

# Test dataloader with 3's only
threes_index = []
nonthrees_index = []
for i in range(0, len(testdata)):
    if testdata[i][1] == 3:
        threes_index.append(i)
    else:
        nonthrees_index.append(i)
three_test_loader = torch.utils.data.DataLoader(testdata, batch_size=64,
            sampler = torch.utils.data.SubsetRandomSampler(threes_index))
nonthree_test_loader = torch.utils.data.DataLoader(testdata, batch_size=64,
            sampler = torch.utils.data.SubsetRandomSampler(nonthrees_index))



# Train dataloaders with limited 3s
nonthrees_index = []
threes_index = []
count = 0
for i in range(0, len(traindata)):
    if traindata[i][1] != 3:
        nonthrees_index.append(i)
        threes_index.append(i)
    if traindata[i][1] == 3 and count < 100:
        count += 1
        threes_index.append(i)
nonthree_train_loader = torch.utils.data.DataLoader(traindata, batch_size=64,
                   sampler = torch.utils.data.SubsetRandomSampler(nonthrees_index))
three_train_loader = torch.utils.data.DataLoader(traindata, batch_size=64,
                   sampler = torch.utils.data.SubsetRandomSampler(threes_index))

# Unlearning dataset with all "3" labels randomly assigned
unlearningdata = copy.deepcopy(traindata)
unlearninglabels = list(range(10))
unlearninglabels.remove(3)
for i in range(len(unlearningdata)):
    if unlearningdata.targets[i] == 3:
        unlearningdata.targets[i] = random.choice(unlearninglabels)
unlearning_train_loader = torch.utils.data.DataLoader(unlearningdata, batch_size=64, shuffle=True)

# Get the folders to save the results in
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
results_folder = os.path.join(project_root, "results", "amnesiac")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


# Hyperparameters
batch_size_train = 64
batch_size_test = 64
log_interval = 16
num_classes = 10
torch.backends.cudnn.enabled = True
criterion = F.nll_loss


trainingepochs = 10
forgetfulepochs = 10
naive_accuracy_three = []
naive_accuracy_nonthree = []

# load resnet 18 and change to fit problem dimensionality
resnet = models.resnet18()
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
resnet.fc = nn.Sequential(nn.Linear(512, num_classes), nn.LogSoftmax(dim=1))
resnet = resnet.to(device)
optimizer = optim.Adam(resnet.parameters())




# ================================================
# Amnesiac unlearning
# ================================================


from src.modelling.amnesiac import AmnesiacTrainer

# First we train the model for trainingepochs epochs
trainer = AmnesiacTrainer(model=resnet, criterion=criterion, optimizer=optimizer, device=device, cache_gradients=False)

trainer.train(three_train_loader, 
              val_loader=None,
              epochs=trainingepochs, 
              class_to_forget=3,
              save_accuracy_to_file=True,
              save_accuracy_to_file_name="before_forgetting_threes",
              retain_loader=nonthree_test_loader,
              forget_loader=three_test_loader
              )

trainer.test(all_data_test_loader, dname="Before forgetting, All data", save_to_file_name="before_forgetting_all_data")
trainer.test(three_test_loader, dname="Before forgetting, Threes", save_to_file_name="before_forgetting_threes")
trainer.test(nonthree_test_loader, dname="Before forgetting, Nonthrees", save_to_file_name="before_forgetting_nonthrees")

trainer.forget(indices_to_forget=None)

trainer.test(all_data_test_loader, dname="After forgetting, All data", save_to_file_name="after_forgetting_all_data")
trainer.test(three_test_loader, dname="After forgetting, Threes", save_to_file_name="after_forgetting_threes")
trainer.test(nonthree_test_loader, dname="After forgetting, Nonthrees", save_to_file_name="after_forgetting_nonthrees")

# repair
trainer.train(nonthree_train_loader, 
              val_loader=None, 
              epochs=forgetfulepochs, 
              repair=True,
              save_accuracy_to_file=True,
              save_accuracy_to_file_name="after_repair_nonthrees",
              retain_loader=nonthree_test_loader,
              forget_loader=three_test_loader
              )

trainer.test(all_data_test_loader, dname="After repair, All data", save_to_file_name="after_repair_all_data")
trainer.test(three_test_loader, dname="After repair, Threes", save_to_file_name="after_repair_threes")
trainer.test(nonthree_test_loader, dname="After repair, Nonthrees", save_to_file_name="after_repair_nonthrees")





