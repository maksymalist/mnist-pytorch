import matplotlib.pyplot as plt
from data.dataloader import validation_data
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualise(images, predictions):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 20))
    
    for i in range(len(images)):
        ax[i].imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
        ax[i].set_title(f"Prediction: {predictions[i]}")
        ax[i].axis("off")
        
    plt.show()