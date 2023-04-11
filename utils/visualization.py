import matplotlib.pyplot as plt
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualise(images, predictions):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 20))
    
    for i in range(len(images)):
        ax[i].imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
        ax[i].set_title(f"Prediction: {predictions[i]}")
        ax[i].axis("off")
        
    plt.show()

def metrics_visualisation(loss, accuracy, epochs):

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].plot(epochs, loss)
    ax[0].set_title("Loss over time")

    ax[1].plot(epochs, accuracy)
    ax[1].set_title("Accuracy over time")

    plt.show()




