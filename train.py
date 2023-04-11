from model import Megatron3000
from data.dataloader import train_loader, val_loader
from utils.visualization import metrics_visualisation
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
SAVE_PATH = os.path.join(os.getcwd(), "model/model.pth")

model = Megatron3000().to(DEVICE)
learning_rate = 0.01
params = model.parameters()
optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.1, verbose=True)

def compute_accuracy(y_hat, target):
    return (y_hat.argmax(dim=1) == target).float().mean() * 100

def train():
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(EPOCHS):
        losses = []
        accuracies = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            y_hat = model(data)

            loss = F.cross_entropy(y_hat, target)
            losses.append(loss.item())
            loss.backward()
            
            accuracy = compute_accuracy(y_hat, target)
            accuracies.append(accuracy)

            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * 100}/60000]  Loss: {loss.item():.6f} Accuracy: {accuracy}")

        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(accuracies) / len(accuracies)
        epoch_losses.append(mean_loss)
        epoch_accuracies.append(mean_accuracy)
        scheduler.step(mean_loss)

    metrics_visualisation(epoch_losses, epoch_accuracies, [i+1 for i in range(EPOCHS)])

def validate():
    
    print("\n")             
    print("##################")
    print("### Validation ###")
    print("##################")
    print("\n") 

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            y_hat = model(data)
            loss = F.cross_entropy(y_hat, target)
            
            if batch_idx % 100 == 0:
                print(f"Validation [{batch_idx * 100}/10000]  Loss: {loss.item():.6f} Accuracy: {compute_accuracy(y_hat, target):.2f}%")

def save():
    torch.save(model.state_dict(), SAVE_PATH)
    
train()
validate()
save()
