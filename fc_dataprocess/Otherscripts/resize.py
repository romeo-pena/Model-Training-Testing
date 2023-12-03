import torch
import os
from tqdm import tqdm
import random
from style_utils import imshow, run_style_transfer
import time
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import warnings

warnings.filterwarnings("ignore")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = (1024, 2048)  # use small size if no gpu
loader = transforms.Compose([
    transforms.Resize(imsize)])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image)
    return image


# load images

rgb = os.listdir("rgb")
rgb = list(rgb)
rgb = tqdm(rgb)

outputdir = os.listdir("resized")
outputdir = list(outputdir)
# print(file not in )

for file in rgb:
    if file not in outputdir:
        content_path = "rgb/" + file
        content_img = image_loader(content_path)

        output_path = "resized/" + file
        content_img.save(output_path)
    time.sleep(0.05)
