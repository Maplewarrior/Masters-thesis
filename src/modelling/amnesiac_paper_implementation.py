# Code adapted from https://github.com/lmgraves/AmnesiacML/blob/main/Amnesiac%20Unlearning%20-%20MNIST.ipynb

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from scipy import ndimage
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

# Download MNIST into the writable directory
traindata = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
testdata = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)


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

steps_folder = os.path.join(results_folder, "steps")
if not os.path.exists(steps_folder):
    os.makedirs(steps_folder)

accuracy_folder = os.path.join(results_folder, "accuracy") 
if not os.path.exists(accuracy_folder):
    os.makedirs(accuracy_folder)

resnet_path = os.path.join(results_folder, "resnet.pt")
if not os.path.exists(os.path.dirname(resnet_path)):
    os.makedirs(os.path.dirname(resnet_path))

# Hyperparameters
batch_size_train = 64
batch_size_test = 64
log_interval = 16
num_classes = 10
torch.backends.cudnn.enabled = True
criterion = F.nll_loss


# ================================================
# Training method
# ================================================

def train(model, epoch, loader, returnable=False):
    model.train()
    model = model.to(device)
    if returnable:
        thracc = []
        nacc = []
        batches = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        steps = []
        if 3 in target:
            before = {}
            for param_tensor in model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    before[param_tensor] = model.state_dict()[param_tensor].clone()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if 3 in target:
            batches.append(batch_idx)
            after = {}
            for param_tensor in model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    after[param_tensor] = model.state_dict()[param_tensor].clone()
            step = {}
            for key in before:
                step[key] = after[key] - before[key]
                f = open(f"{steps_folder}/e{epoch}b{batches[-1]:04}.pkl", "wb")
                pickle.dump(step, f)
                f.close()
        if batch_idx % log_interval == 0:
            print("\rEpoch: {} [{:6d}]\tLoss: {:.6f}".format(
                epoch, batch_idx*len(data), loss.item()), end="")
        if returnable and batch_idx % 10 == 0:
            thracc.append(test(model, three_test_loader, dname="Threes only", printable=False))
            if batch_idx % 10 == 0:
                nacc.append(test(model, nonthree_test_loader, dname="nonthree only", printable=False))
            model.train()
    if returnable:
        return thracc, nacc, batches, steps



# ================================================
# Testing method
# ================================================

def test(model, loader, dname="Test set", printable=True):
    model.eval()
    model = model.to(device)
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += target.size()[0]
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    if printable:
        print('{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            dname, test_loss, correct, total,
            100. * correct / total
            ))
    return 1. * correct / total




# ================================================
# Training and testing
# ================================================

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


# Train new model for 5 epochs
steps = []
for epoch in range(1, trainingepochs+1):
  starttime = time.process_time()
  # train(resnet, epoch, all_data_train_loader, returnable=False)
  thracc, nacc, three_batches, three_steps = train(resnet, epoch, three_train_loader, returnable=True)
  naive_accuracy_three += thracc
  naive_accuracy_nonthree += nacc
  steps = steps + three_steps
  print(f"{three_batches} batches effected")
  test(resnet, all_data_test_loader, dname="All data")
  test(resnet, three_test_loader, dname="Threes  ")
  test(resnet, nonthree_test_loader, dname="Nonthree")
  print(f"Time taken: {time.process_time() - starttime}")
  path = F"{resnet_path}/selective_trained_e{epoch}.pt"
  torch.save({
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)
  path = F"{accuracy_folder}/mnist_selective_trained_accuracy_three_e{epoch}.txt"
  with open(path, 'w') as f:
    for data in naive_accuracy_three:
      f.write(f"{data},")
  path = F"{accuracy_folder}/mnist_selective_trained_accuracy_nonthree_e{epoch}.txt"
  with open(path, 'w') as f:
    for data in naive_accuracy_nonthree:
      f.write(f"{data},")


path = F"{results_folder}/selective_trained.pt"
torch.save({
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)

path = F"{accuracy_folder}/selective_trained_accuracy_three.txt"
with open(path, 'w') as f:
  for data in naive_accuracy_three:
    f.write(f"{data},")

path = F"{accuracy_folder}/selective_trained_accuracy_nonthree.txt"
with open(path, 'w') as f:
  for data in naive_accuracy_nonthree:
    f.write(f"{data},")

path = F"{results_folder}/selective_trained.pt"
checkpoint = torch.load(path)
resnet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for i in range(1, trainingepochs+1):
    for j in range(1600):
        path = f"steps/e{i}b{j:04}.pkl"
        try:
            # print("before")
            f = open(path, "rb")
            steps = pickle.load(f)
            f.close()
            print(f"\rLoading steps/e{i}b{j:04}.pkl", end="")
            const = 1
            with torch.no_grad():
                state = resnet.state_dict()
                for param_tensor in state:
                    if "weight" in param_tensor or "bias" in param_tensor:
                      state[param_tensor] = state[param_tensor] - const*steps[param_tensor]
            resnet.load_state_dict(state)
        except:
            # print(f"\r{i},{j}", end="")
            pass


test(resnet, all_data_test_loader, dname="All data")
test(resnet, three_test_loader, dname="Threes  ")
test(resnet, nonthree_test_loader, dname="Nonthree")


path = F"{resnet_path}/selective_post_trained.pt"
torch.save({
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)

path = F"{results_folder}/selective_post_trained.pt"
checkpoint = torch.load(path)
resnet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


selective_post_accuracy_three = []
selective_post_accuracy_nonthree =[]


# Train model for 10 forgetful epochs
for epoch in range(trainingepochs+1,trainingepochs+forgetfulepochs+1):
  # train(resnet, epoch, nonthree_train_loader, returnable=False)
  thracc, nacc, _, _ = train(resnet, epoch, nonthree_train_loader, returnable=True)
  selective_post_accuracy_three += thracc
  selective_post_accuracy_nonthree += nacc
  test(resnet, all_data_test_loader, dname="All data")
  test(resnet, three_test_loader, dname="Threes  ")
  test(resnet, nonthree_test_loader, dname="Nonthree")
  path = F"{resnet_path}/selective-post-epoch-{epoch}.pt"
  torch.save({ 
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)
  path = F"{accuracy_folder}/selective_post_accuracy_three_e{epoch}.txt"
  with open(path, 'w') as f:
    for data in naive_accuracy_three:
      f.write(f"{data},")
  path = F"{accuracy_folder}/selective_post_accuracy_nonthree_e{epoch}.txt"
  with open(path, 'w') as f:
    for data in naive_accuracy_nonthree:
      f.write(f"{data},")

path = F"{accuracy_folder}/selective_post_accuracy_three.txt"
with open(path, 'w') as f:
  for data in selective_post_accuracy_three:
    f.write(f"{data},")

path = F"{accuracy_folder}/selective_post_accuracy_nonthree.txt"
with open(path, 'w') as f:
  for data in selective_post_accuracy_nonthree:
    f.write(f"{data},")


if __name__ == "__main__":
    # Example usage
    for images, labels in all_data_train_loader:
        print("Batch shape:", images.shape)
        print("Labels shape:", labels.shape)
        break