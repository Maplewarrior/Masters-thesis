import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from src.modelling.maso import maso_dataset, MasoDataset, MASO
from src.modelling.trainer import Trainer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
n_samples = 500
n_features = 2
n_classes = 2
seed = 42

class NeuralNet(nn.Module):
    def __init__(self, M: int, n_classes: int, width: int = 8) -> None:
        super().__init__()
        """
            M (int): Feature dimension of the data
            n_classes (int): Number of classes in the dataset.
        """
        self.M = M
        self.n_classes = n_classes
        self.net = nn.Sequential(nn.Linear(self.M, self.n_classes))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, start_idx: int = 0, stop_idx: int = None):
        logits = self.net[start_idx:stop_idx](x)
        return {'logits': logits, 'probabilities': self.softmax(logits), 'predictions': torch.argmax(logits, dim=-1)}

    def inference(self, x, start_idx: int = 0, stop_idx: int = None):
        self.eval()
        with torch.no_grad():
            return self(x, start_idx, stop_idx)

np.random.seed(seed)

X, y = maso_dataset(n_samples=n_samples, 
                    n_features=n_features, 
                    n_classes=n_classes)

# Train/val/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

# Since y is already one-hot encoded from maso_dataset
train_dataset = MasoDataset(X_train, y_train, onehot_labels=True, n_classes=n_classes)
val_dataset = MasoDataset(X_val, y_val, onehot_labels=True, n_classes=n_classes)
test_dataset = MasoDataset(X_test, y_test, onehot_labels=True, n_classes=n_classes)


# Convert to DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
# ==============================
# Create model
# ==============================
# ! THIS EXPECTS A ONE LAYER MODEL
model = NeuralNet(M=n_features, n_classes=n_classes)

maso = MASO(model, train_loader, _lambda=0.3)

trainer = Trainer(model, 
                     train_dataloader=train_loader, 
                     val_dataloader=val_loader, 
                     lr=0.001, 
                     device='cpu')
    
trainer.train(n_epochs=60, save_weights=True)
trainer.eval()

weights = maso.extract_maso_params()

if len(y.shape) == 2 and y.shape[1] > 1:
    y = np.argmax(y, axis=1)
classes = np.unique(y)
colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
custom_cmap = ListedColormap(colors)

# Create a grid of points
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, n_samples),
                       np.linspace(x2_min, x2_max, n_samples))

# Plot the weights
plt.figure(figsize=(10, 5))
W, b = weights[-1]

# Calculate spline values for each grid point
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
spline_values = np.zeros((len(grid_points), len(W)))

for i in range(len(W)):
    # We calculate the value of a grid point using a spline
    values = -(W[i, 0] * grid_points[:, 0] + b[i]) / W[i, 1] - grid_points[:, 1]
    # "Make" spline via ReLU
    spline_values[:, i] = torch.relu(torch.tensor(values, dtype=torch.float32)).numpy()

# Create a categorical array based on which splines are active
# 0: no splines active
# 1: only first spline active
# 2: only second spline active
# 3: both splines active
categorical = np.zeros(len(grid_points))
for i in range(len(grid_points)):
    if spline_values[i, 0] > 0 and spline_values[i, 1] > 0:
        categorical[i] = 3
    elif spline_values[i, 0] > 0:
        categorical[i] = 1
    elif spline_values[i, 1] > 0:
        categorical[i] = 2

categorical = categorical.reshape(xx1.shape)

# Create custom colormap for the four categories
region_colors = ['white', 'lightblue', 'lightgreen', 'orange']
region_cmap = ListedColormap(region_colors)

# Create the plot
plt.contourf(xx1, xx2, categorical, levels=4, cmap=region_cmap, alpha=0.7)

# Add the parameterized lines
x = np.linspace(x1_min, x1_max, n_samples)
for i in range(len(W)):
    y = -(W[i, 0] * x + b[i]) / W[i, 1]
    plt.plot(x, y, linewidth=2, label=f'w{i}', color=colors[i])

plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap=custom_cmap, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('MASO Active Regions\nWhite: None, Blue: Spline 1, Green: Spline 2, Orange: Both')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.legend()
plt.show()






