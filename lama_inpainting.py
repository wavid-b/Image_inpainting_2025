import numpy as np
from PIL import Image
from lama_cleaner.model_manager import ModelManager
import torch
import tqdm
from lama_cleaner.schema import Config, HDStrategy, LDMSampler 

#set up config
config = Config(
    model="ldm",           # LaMa erase model (others: ldm, zits, mat, fcf, manga…) :contentReference[oaicite:3]{index=3}
    device="cuda",          # or "cpu"
    no_half=False,          # allow use of fp16 for speed (if supported)
    # --- required for any LDM-based model ---
    ldm_steps=20,                             # number of diffusion steps
    ldm_sampler=LDMSampler.plms,              # defaults to "plms" if you omit

    # --- required for the High-Res preprocessing strategy ---
    hd_strategy=HDStrategy.ORIGINAL,          # ORIGINAL, RESIZE or CROP
    hd_strategy_crop_margin=32,               # only for CROP/RESIZE modes
    hd_strategy_crop_trigger_size=512,        # threshold for switching to crop
    hd_strategy_resize_limit=512,             # max side length before resizing
    # you can also set: enable_controlnet=False, controlnet_args=…
)
images = np.load("mnist_files/small_test_true.npy")  # shape (N, 28, 28)
masks  = np.load("mnist_files/small_test_mask.npy")   # shape (N, 28, 28)

# --------- Ensure data is in correct format -----------
# Convert to uint8 if necessary
if images.max() <= 1.0:
    images = (images * 255).astype(np.uint8)

if masks.max() <= 1.0:
    masks = (masks * 255).astype(np.uint8)

# --------- Initialize LaMa model -----------
model = ModelManager(name="ldm", device="cuda" if torch.cuda.is_available() else "cpu")

# --------- Process each image -----------
inpainted_images = []

for i in tqdm.trange(len(images), desc="Inpainting"):
    img_gray = images[i]
    mask = masks[i]

    # Convert grayscale to RGB (required by model)
    img_rgb = np.stack([img_gray]*3, axis=-1)  # shape (28, 28, 3)

    # Run LaMa inpainting
    result_rgb = model(img_rgb, mask, config)  # result is RGB

    # Convert back to grayscale
    result_gray = np.mean(result_rgb, axis=-1).astype(np.uint8)  # shape (28, 28)

    inpainted_images.append(result_gray)
    
    #output progress 
    if i % 25 == 0: 
        print(f"Finished image {i}")

# Convert to NumPy array
inpainted_images = np.stack(inpainted_images, axis=0)  # shape (N, 28, 28)

# --------- Save (optional) -----------
np.save("mnist_files/ldm_inpainted_images.npy", inpainted_images)