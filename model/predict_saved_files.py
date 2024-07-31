import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import json
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
    
    #crop to middle 512 x 512
    width, height = image.size
    left = (width - 512) // 2
    top = (height - 512) // 2
    right = left + 512
    bottom = top + 512
    image_cropped = image.crop((left, top, right, bottom))
    image_cropped = transforms.ToTensor()(image_cropped).unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass through the model
    with torch.no_grad():
        output = model(image_cropped)

    # Convert the output to an image format (assuming the output is a single-channel image)
    output_image = output.squeeze(0).detach().numpy()

    # Handle different number of channels
    if output_image.shape[0] == 1:  # Single-channel (grayscale)
        output_image = output_image[0]  
        output_image = output_image.astype(np.float32)

        #save output as .npy file
        npy_output_file = get_unique_filename("/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/output_image.npy")
        np.save(npy_output_file, output_image)
        print(f"saved as model output image file {npy_output_file}")

        print(np.unique(output_image))

        # Apply colormap to the grayscale image
        colormap_image = cv2.applyColorMap((output_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save the colormapped image as .png file
        png_output_file = npy_output_file.replace(".npy", ".png")
        cv2.imwrite(png_output_file, colormap_image)
        print(f"saved as model png file: {png_output_file}")

    elif output_image.shape[0] == 3:  # RGB image
        output_image = output_image.transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        output_image = (output_image * 255).astype(np.uint8)
        output_image_pil = Image.fromarray(output_image, mode='RGB')
        
        # Save the output image
        output_image_path = '/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/output_image.png'
        output_image_pil.save(get_unique_filename(output_image_path))
        print(f"Output image saved to {output_image_path}")

    dbh_segmentation = np.load(dbh_file)
    height, width = dbh_segmentation.shape
    left = (width - 512) // 2
    top = (height - 512) // 2
    right = left + 512
    bottom = top + 512
    dbh_segmentation_cropped = dbh_segmentation[top:bottom, left:right]

    print(np.unique(dbh_segmentation_cropped))

    # Apply colormap to the cropped DBH segmentation for visualization
    colormap_dbh_segmentation = cv2.applyColorMap((dbh_segmentation_cropped * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Save the colormapped DBH segmentation image
    colormap_dbh_segmentation_filename = get_unique_filename('/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/colormap_dbh_segmentation.png')
    cv2.imwrite(colormap_dbh_segmentation_filename, colormap_dbh_segmentation)
    print(f"saved colormap_dbh_segmentation.png as {colormap_dbh_segmentation_filename}")

    # Calculate accuracy
    if output_image.shape == dbh_segmentation_cropped.shape:
        accuracy = np.mean(np.round(output_image, 1) == np.round(dbh_segmentation_cropped, 1))
        print(f"Accuracy: {accuracy}")
    else:
        print(f"Shape mismatch: output_image shape {output_image.shape} vs dbh_segmentation_cropped shape {dbh_segmentation_cropped.shape}")

    # Display the base cropped DBH colormapping and the predicted colormapping side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(colormap_dbh_segmentation)
    axs[0].set_title('Cropped DBH Colormapping')
    axs[0].axis('off')

    axs[1].imshow(colormap_image)
    axs[1].set_title('Predicted Colormapping')
    axs[1].axis('off')

    # Save the combined image as a PNG file
    combined_image_filename = get_unique_filename('/Users/michellechen/cs/other/research/nerfforest-model/model/datafiles/combined_colormapping.png')
    plt.savefig(combined_image_filename)
    print(f"saved combined colormapping as {combined_image_filename}")
    plt.close()