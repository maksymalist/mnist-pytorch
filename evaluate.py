from data.dataloader import validation_data
from model import Megatron3000
from utils.visualization import visualise
import torch
import os

SAVE_PATH = os.path.join(os.getcwd(), "model/model.pth")

model = Megatron3000()
model.load_state_dict(torch.load(SAVE_PATH))

def predict(image):
    with torch.no_grad():
        return model(image).argmax().item()
    
images = []
predictions = []

for i in range(10):
    images.append(validation_data[i][0])
    predictions.append(predict(validation_data[i][0].unsqueeze(0)))
    

    
visualise(images, predictions)

