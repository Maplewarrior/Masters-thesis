from src.modelling.neural_network import SimpleNeuralNet
from src.modelling.trainer import Trainer
from src.modelling.scrub import ScrubR

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy 
import pdb


def eval(model, dataloader, criterion):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for ipt, label in dataloader:
            ipt = ipt.to(device)
            label = label.to(device)
            
            out = model.inference(ipt)
            loss = criterion(out['logits'], label)
            
            total_loss += loss.item()
            total_correct += ((out['probabilities'].argmax(dim=1)) == label.argmax(dim=1)).sum().item()
            total_samples += len(label)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

M = 2
n_classes = 3
device = 'cpu'
criterion = torch.nn.CrossEntropyLoss()

class_0_idxs = (0, 333)
class_1_idxs = (360, 667)
class_2_idxs = (667, 1000)
torch.manual_seed(50)
X = torch.randn(size=(1000, 2)) * 0.1


# for i, idxs in enumerate([class_0_idxs, class_1_idxs, class_2_idxs]):
#     index = torch.tensor(list(range(idxs[0], idxs[1])))
#     X.index_add(dim=0, index=index, source=torch.zeros_like(index) + 1.5)
X[:333, 0] += 1.5
X[:333, 1] += 1.7

X[333:667, 0] += 0.5
X[333:667, 1] += 1.5

# outliers
X[333:360, 1] -= 0.7

y = torch.zeros((1000, 3))
y[:333, 0] = 1
y[333:360:, 2] = 1 # outliers
y[360:667, 1] = 1
y[667:, 2] = 1


X_val = torch.randn(size=(300, 2)) * 0.1
X_val[:100, 0] += 1.5
X_val[:100, 1] += 1.7

X_val[100:200, 0] += 0.5
X_val[100:200, 1] += 1.5
X_val[200:300, 1] -= 0.7

y_val = torch.zeros((300, 3))
y_val[:100, 0] = 1
y_val[100:200, 1] = 1
y_val[200:300, 2] = 1

colors = ['red', 'blue', 'green']

for i, idxs in enumerate([class_0_idxs, class_1_idxs, class_2_idxs]):
    vals = X.index_select(dim=0, index=torch.tensor(list(range(idxs[0], idxs[1]))))
    plt.scatter(vals[:,0], vals[:, 1], c=colors[i], label=f'class {i+1}')

plt.scatter(X[333:360, 0], X[333:360, 1], c=colors[-1], label='outliers', marker='d')
plt.legend()
plt.show()



X_retain = torch.cat((X[:333, :], X[360:, :]), dim=0)
X_forget = X[333:360, :] 

y_retain = torch.cat((y[:333, :], y[360:, :]), dim=0)
y_forget = y[333:360, :] 

train_loader = DataLoader(TensorDataset(X, y), batch_size=8)
retain_loader = DataLoader(TensorDataset(X_retain, y_retain), batch_size=8)
forget_dataset = TensorDataset(X_forget, y_forget)
forget_dataset.X = X_forget
forget_dataset.y = y_forget
forget_loader = DataLoader(forget_dataset, batch_size=3)
val_dataset = TensorDataset(X_val, y_val)
val_dataset.X = X_val
val_dataset.y = y_val

val_loader = DataLoader(val_dataset, batch_size=4)



retrained_model = SimpleNeuralNet(M, n_classes)
unlearned_model = SimpleNeuralNet(M, n_classes)

# train unlearned/original model
trainer = Trainer(unlearned_model, train_loader, val_loader)
trainer.train()
original_model = copy.deepcopy(unlearned_model)

# train retrained model
trainer = Trainer(retrained_model, retain_loader, val_loader)
trainer.train()

scrub_r = ScrubR(unlearned_model, original_model, alpha=1., gamma=1.)
scrub_r(retain_loader, forget_loader, val_loader, n_rounds=10)



retrained_model_val = eval(retrained_model, val_loader, criterion)
retrained_model_forget = eval(retrained_model, forget_loader, criterion)
retrained_model_retain = eval(retrained_model, retain_loader, criterion)
print(f'\n\nretrained model on validation set: loss={retrained_model_val[0]:.4f}, acc={retrained_model_val[1]:.4f}')
print(f'retrained model on forget set: loss={retrained_model_forget[0]:.4f}, acc={retrained_model_forget[1]:.4f}')
print(f'retrained model on retain set: loss={retrained_model_retain[0]:.4f}, acc={retrained_model_retain[1]:.4f}')

unlearned_model_val = eval(unlearned_model, val_loader, criterion)
unlearned_model_forget = eval(unlearned_model, forget_loader, criterion)
unlearned_model_retain = eval(unlearned_model, retain_loader, criterion)
print(f'\nunlearned model on validation set: loss={unlearned_model_val[0]:.4f}, acc={unlearned_model_val[1]:.4f}')
print(f'unlearned model on forget set: loss={unlearned_model_forget[0]:.4f}, acc={unlearned_model_forget[1]:.4f}')
print(f'unlearned model on retain set: loss={unlearned_model_retain[0]:.4f}, acc={unlearned_model_retain[1]:.4f}\n\n')

pdb.set_trace()