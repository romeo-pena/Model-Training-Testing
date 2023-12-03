import torch
import os
from tqdm import tqdm
import time
from PIL import Image

import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = (256, 512)  # use small size if no gpu
loader = transforms.Compose([
    transforms.Resize(imsize)])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image)
    return image


# load images

rgb = os.listdir("images")
rgb = list(rgb)
rgb = tqdm(rgb)

outputdir = os.listdir("resized_images")
outputdir = list(outputdir)
# print(file not in )

for file in rgb:
    if file not in outputdir:
        content_path = "images/" + file
        content_img = image_loader(content_path)

        output_path = "resized_images/" + file
        content_img.save(output_path)
    time.sleep(0.05)
