import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from unet import UNet, image_path
from model import UNET
from train import get_unique_filename
import cv2

model_path = "/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/models/model_16.pth"
dbh_file = "/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/dbh.npy"
image_path = "/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/DJI_0002.JPG"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))

    image = Image.open(image_path)
    
    # Crop to middle 512 x 512
    width, height = image.size
    left = (width - 512) // 2
    top = (height - 512) // 2
    right = left + 512
    bottom = top + 512
    image = image.crop((left, top, right, bottom))
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)

    print(np.unique(output))

    # Convert the output to an image format (assuming the output is a single-channel image)
    output_image = output.squeeze(0).detach().cpu().numpy()

    # Handle different number of channels
    if output_image.shape[0] == 1:  # Single-channel (grayscale)
        output_image = output_image[0]  
        output_image = output_image.astype(np.float32)

    elif output_image.shape[0] == 3:  # RGB image
        output_image = output_image.transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        output_image = (output_image * 255).astype(np.uint8)

    dbh_segmentation = np.load(dbh_file)
    height, width = dbh_segmentation.shape
    left = (width - 512) // 2
    top = (height - 512) // 2
    right = left + 512
    bottom = top + 512
    dbh_segmentation_cropped = dbh_segmentation[top:bottom, left:right]

    # Normalize the output_image to match the range of dbh_segmentation_cropped
    dbh_min, dbh_max = np.min(dbh_segmentation_cropped), np.max(dbh_segmentation_cropped)
    output_image_min, output_image_max = output_image.min(), output_image.max()
    normalized_output_image = ((output_image - output_image_min) / (output_image_max - output_image_min)) * (dbh_max - dbh_min) + dbh_min

    # Apply colormap to the cropped DBH segmentation for visualization
    colormap_dbh_segmentation = cv2.applyColorMap((dbh_segmentation_cropped * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Apply the same colormap to the model output image
    colormap_output_image = cv2.applyColorMap((output_image * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Apply colormap to the normalized output image
    colormap_output_image_normalized = cv2.applyColorMap((normalized_output_image * 255 / dbh_max).astype(np.uint8), cv2.COLORMAP_JET)

    # Calculate accuracy
    if output_image.shape == dbh_segmentation_cropped.shape:
        accuracy = np.mean(np.round(normalized_output_image, 1) == np.round(dbh_segmentation_cropped, 1))
        print(f"Accuracy: {accuracy}")
    else:
        print(f"Shape mismatch: output_image shape {output_image.shape} vs dbh_segmentation_cropped shape {dbh_segmentation_cropped.shape}")

    # Display the base cropped DBH colormapping and the predicted colormapping side by side
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(colormap_dbh_segmentation)
    axs[0].set_title('Cropped DBH Colormapping')
    axs[0].axis('off')

    axs[1].imshow(colormap_output_image)
    axs[1].set_title('Predicted Colormapping')
    axs[1].axis('off')

    axs[2].imshow(colormap_output_image_normalized)
    axs[2].set_title('Normalized Predicted Colormapping')
    axs[2].axis('off')

    print(f"Output image shape: {output_image.shape}, DBH segmentation shape: {dbh_segmentation_cropped.shape}, normalized output image shape: {normalized_output_image.shape}")
    print(f"output_image unique values: {np.unique(output_image)}, dbh_segmentation_cropped unique values: {np.unique(dbh_segmentation_cropped)}, normalized_output_image unique values: {np.unique(normalized_output_image)}")
    # Show the combined image
    plt.savefig("/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/combined_image.png")
    plt.show()