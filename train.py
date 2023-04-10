from model import Megatron3000
from data.dataloader import train_loader, val_loader
import torch.nn.functional as F
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
SAVE_PATH = os.path.join(os.getcwd(), "model/model.pth")

model = Megatron3000().to(DEVICE)
params = model.parameters()
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)


def accuracy(y_hat, target):
    return (y_hat.argmax(dim=1) == target).float().mean() * 100

def train():
    for epoch in range(EPOCHS):
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            y_hat = model(data)
            loss = F.cross_entropy(y_hat, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * 100}/60000]  Loss: {loss.item():.6f} Accuracy: {accuracy(y_hat, target):.2f}%")

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
                print(f"Validation [{batch_idx * 100}/10000]  Loss: {loss.item():.6f} Accuracy: {accuracy(y_hat, target):.2f}%")

def save():
    torch.save(model.state_dict(), SAVE_PATH)
    
train()
validate()
save()