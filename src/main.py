import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Argument parser for configuration
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['standard', 'odenet'], default='standard')
parser.add_argument('--dataset', type=str, choices=['mnist'], default='mnist')  # Simplified to MNIST for your case
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Data Loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((8, 8))])
train_dataset = datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform)
train_idx = (train_dataset.targets == 0) | (train_dataset.targets == 1)
test_idx = (test_dataset.targets == 0) | (test_dataset.targets == 1)
train_dataset.data, train_dataset.targets = train_dataset.data[train_idx], train_dataset.targets[train_idx] - 0
test_dataset.data, test_dataset.targets = test_dataset.data[test_idx], test_dataset.targets[test_idx] - 0
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Standard Neural Network
class StandardNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=16, output_dim=2):
        super(StandardNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Neural ODE Block
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim)
        )
    
    def forward(self, t, h):
        return self.net(h)

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0., 1.]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, method='dopri5')
        return out[1]

class ODENet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=16, output_dim=2):
        super(ODENet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.odeblock = ODEBlock(ODEFunc(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.odeblock(x)
        x = self.output_layer(x)
        return x

# Training Function
def train(model, train_loader, epochs, model_name):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1} ({model_name}), Loss: {avg_loss:.4f}')
    return losses

# Accuracy Function
def test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, all_preds, all_labels

# Visualization Function
def visualize_results(model_name, train_losses, train_acc, test_acc, preds, labels):
    plt.figure(figsize=(12, 4))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png')
    plt.close()

# Main Execution
results = {}
for model_type in ['standard', 'odenet']:
    args.model = model_type
    if args.model == 'standard':
        model = StandardNet().to(device)
    else:
        model = ODENet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f'\nTraining {args.model} on {args.dataset}')
    train_losses = train(model, train_loader, args.epochs, args.model)
    train_acc, train_preds, train_labels = test_accuracy(model, train_loader)
    test_acc, test_preds, test_labels = test_accuracy(model, test_loader)
    print(f'Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    results[model_type] = {'losses': train_losses, 'train_acc': train_acc, 'test_acc': test_acc,
                          'train_preds': train_preds, 'train_labels': train_labels,
                          'test_preds': test_preds, 'test_labels': test_labels}
    
    visualize_results(args.model, train_losses, train_acc, test_acc, test_preds, test_labels)

# Part C: Comparison
print("\nPart C: Performance Comparison")
for model_type in results:
    print(f"{model_type.capitalize()} - Train Acc: {results[model_type]['train_acc']:.4f}, "
          f"Test Acc: {results[model_type]['test_acc']:.4f}, Final Loss: {results[model_type]['losses'][-1]:.4f}")

# Discussion (Part C(b))
print("\nDiscussion: The Neural ODE model uses continuous depth, allowing it to adapt dynamically to data changes "
      "over time, potentially improving generalization. However, it may require more computational resources "
      "compared to the discrete StandardNet, which relies on fixed layers. The high accuracy in both models "
      "suggests the task (binary MNIST) is simple enough for both approaches, but ODE's behavior might shine "
      "with more complex datasets or longer training.")
