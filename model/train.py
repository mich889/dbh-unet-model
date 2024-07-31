import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from unet import UNet, image_path, init_weights
from torch.utils.data import DataLoader, Dataset
from model import UNET
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "datafiles/DJI_0002.JPG")
SEGMENTATION_PATH = os.path.join(BASE_DIR, "datafiles/dbh.npy")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "datafiles/models/model.pth")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "datafiles/generated_outputs/dbh_comparison.png")
DBH_FILE = os.path.join(BASE_DIR, "datafiles/dbh.npy")

class CustomDataset(Dataset):
    def __init__(self, image_path, segmentation_path, transform=None):
        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        segmentation = np.load(self.segmentation_path)
        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)
        return image, segmentation

def batch_load_data(image_path, segmentation_path, device):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(image_path, segmentation_path, transform)
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    image_tensors, segmentation_tensors = [], []
    for images, segmentations in dataloader:
        images = images.to(device)
        segmentations = segmentations.to(device)
        image_tensors.append(images)
        segmentation_tensors.append(segmentations)
    image_tensors = torch.cat(image_tensors, dim = 0)
    segmentation_tensors = torch.cat(segmentation_tensors, dim = 0)
    return image_tensors, segmentation_tensors

def load_data(image_path, segmentation_path):
    image = Image.open(image_path).convert("RGB")
    segmentation = np.load(segmentation_path)

    #crop to middle 512 x 512
    width, height = image.size
    left = (width - 512) // 2
    top = (height - 512) // 2
    right = left + 512
    bottom = top + 512
    image = image.crop((left, top, right, bottom))
    segmentation = segmentation[top:bottom, left:right]
    print(image.size, segmentation.shape)
    
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    segmentation_tensor = transform(segmentation).unsqueeze(0).to(device)
    #normalization from segmentation to image
    segmentation_tensor = (segmentation_tensor - segmentation_tensor.min()) / (segmentation_tensor.max() - segmentation_tensor.min())
    image_tensor = transform(image).unsqueeze(0).to(device)

    return image_tensor, segmentation_tensor

def get_unique_filename(filepath):
    if not os.path.exists(filepath):
        return filepath
    filename, file_extension = os.path.splitext(filepath)
    i = 1
    while os.path.exists(filepath):
        filepath = f"{filename}_{i}{file_extension}"
        i += 1
    return filepath
    
def train(model, image_tensor, segmentation_tensor, num_epochs=20):
    #batch dimension aligmnet
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if segmentation_tensor.dim() == 3:
        segmentation_tensor = segmentation_tensor.unsqueeze(0)

    # Convert tensors to the same data type (float32)
    image_tensor = image_tensor.to(torch.float32)
    segmentation_tensor = segmentation_tensor.to(torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    image_tensor = image_tensor.to(device)
    segmentation_tensor = segmentation_tensor.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    print("training started")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(image_tensor)
        loss = criterion(outputs, segmentation_tensor)
        # print(segmentation_tensor, outputs )
        # print(segmentation_tensor.shape, outputs.shape )
        # print(torch.unique(outputs))

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model, losses


def segmentation_to_dbh(segmentation_path, ids_to_labels, output_path):
    segmentation = np.load(segmentation_path)
    with open(ids_to_labels) as f:
        ids_to_labels = json.load(f)
    dbh = np.copy(segmentation)
    ids_to_labels["-1"] = 0
    for original, new in ids_to_labels.items():
        try:
            dbh[segmentation == int(original)] = new
        except:
            dbh[segmentation == int(original)] = 0
    np.save(get_unique_filename(output_path), dbh)
    print(np.unique(dbh))

def plot_training_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    # Load the data
    image_tensor, segmentation_tensor = load_data(IMAGE_PATH, DBH_FILE)

    print("image tensor shape: ", image_tensor.shape)
    print("segmentation tensor shape: ", segmentation_tensor.shape)

    # Initialize the model
    model = UNet().to(device)
    model.apply(init_weights)

    # Train the model
    trained_model, losses = train(model, image_tensor, segmentation_tensor)

    # Save the model
    model_path = get_unique_filename(MODEL_SAVE_PATH)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device)).cpu().numpy()

    print(f"Model output min: {output.min()}, Model output max: {output.max()}")

    # Denormalize the model output to match the DBH range
    dbh_min, dbh_max = np.min(np.load(DBH_FILE)), np.max(np.load(DBH_FILE))
    denormalized_output = output * (dbh_max - dbh_min) + dbh_min
    print(f"(denormalized) Model output min: {denormalized_output.min()}, (denormalized) Model output max: {denormalized_output.max()}")

    # Display the original and predicted DBH side by side
    original_dbh = segmentation_tensor.squeeze().cpu().numpy() * (dbh_max - dbh_min) + dbh_min
    predicted_dbh = denormalized_output.squeeze()
    pred_min, pred_max = np.min(predicted_dbh), np.max(predicted_dbh)
    
    scaled_dbh = ((predicted_dbh - pred_min) * (dbh_max - dbh_min) / (pred_max - pred_min)) + dbh_min


    # Apply colormap to the images (normalized)
    colormap_original_dbh = cv2.applyColorMap((original_dbh * 255 / dbh_max).astype(np.uint8), cv2.COLORMAP_JET)
    colormap_predicted_dbh = cv2.applyColorMap((predicted_dbh * 255 / dbh_max).astype(np.uint8), cv2.COLORMAP_JET)
    scaled_colormap_predicted_dbh = cv2.applyColorMap((scaled_dbh * 255 / dbh_max).astype(np.uint8), cv2.COLORMAP_JET)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(original_dbh, cmap='viridis')
    axs[0].set_title('Original DBH')
    axs[0].axis('off')

    axs[1].imshow(predicted_dbh, cmap='viridis')
    axs[1].set_title('Predicted DBH')
    axs[1].axis('off')

    axs[2].imshow(scaled_dbh, cmap='viridis')
    axs[2].set_title('Scaled Predicted DBH')
    axs[2].axis('off')

    # axs[3].imshow(scaled_dbh, cmap='viridis')
    # axs[3].set_title('Loaded model Scaled Predicted DBH')
    # axs[3].axis('off')

    print(np.unique(original_dbh))
    print(np.unique(predicted_dbh))
    print(np.unique(scaled_dbh))
    print(f"Original DBH min: {original_dbh.min()}, Original DBH max: {original_dbh.max()}", f"Scaled DBH min: {scaled_dbh.min()}")
    print(f"Predicted DBH min: {predicted_dbh.min()}, Predicted DBH max: {predicted_dbh.max()}", f"Scaled DBH max: {scaled_dbh.max()}")

    # Calculate accuracy with threshold
    threshold = 0.1 
    accuracy = np.mean(np.abs(predicted_dbh - original_dbh) < threshold)
    print(f"Accuracy: {accuracy:.2f}%")

    plt_filename = "/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/generated_outputs/dbh_comparison.png"
    plt.savefig(get_unique_filename(plt_filename))
    plt.show()

    plot_training_loss(losses)

    


