import torch
from networks import PConvUNet
import torchvision.transforms as T
import numpy as np
from PIL import Image

# Convert images and masks to torch tensors
def prepare_data(images_np, masks_np):
    # Ensure images are normalized to [0, 1] range
    images_tensor = torch.tensor(images_np, dtype=torch.float32) / 255.0
    masks_tensor = torch.tensor(masks_np, dtype=torch.float32)

    # Add channel dimension (N, 1, 28, 28)
    images_tensor = images_tensor.unsqueeze(1)
    masks_tensor = masks_tensor.unsqueeze(1)

    return images_tensor, masks_tensor

# Load the model
model = PConvUNet()
model.load_state_dict(torch.load('logs/pconv_imagenet/pytorch_model.pth'))
model.eval().cuda()

#import the data from files 
images_np = np.load("mnist_files/mnist_corrupted_test.npy")
masks_np = np.load("mnist_files/true_masks_test.npy")

# Prepare data
images_tensor, masks_tensor = prepare_data(images_np, masks_np)

# Move tensors to GPU if available
images_tensor = images_tensor.cuda()
masks_tensor = masks_tensor.cuda()

# Run inpainting for the entire batch
with torch.no_grad():
    inpainted_images, _ = model(images_tensor * (1 - masks_tensor), masks_tensor)

# Convert inpainted images back to PIL for saving or visualization
inpainted_images_np = (inpainted_images.cpu().numpy() * 255).astype(np.uint8)

# Remove the single channel dimension to get (N, 28, 28)
inpainted_images_np = np.squeeze(inpainted_images_np, axis=1)
print(f"final shape: {inpainted_images_np}")
np.save('mnist_files/PConvUNet_inpainted_images.npy', inpainted_images_np)
