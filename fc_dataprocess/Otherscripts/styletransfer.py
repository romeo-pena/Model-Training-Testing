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
imsize = (256, 512)
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


# Importing the Model
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def image_loader(image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# load images


unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()

cityscapesimages = os.listdir("images")
cityscapesimages = list(cityscapesimages)

rgb = os.listdir("rgb")
rgb = list(rgb)
rgb = tqdm(rgb)

outputdir = os.listdir("style_transfer")
outputdir = list(outputdir)
# print(file not in )

for file in rgb:
    if file not in outputdir:
        chosen = random.choice(cityscapesimages)
        style_path = "images/" + chosen     # Target Style
        content_path = "rgb/" + file    # Source Style
        style_img = image_loader(style_path)
        content_img = image_loader(content_path)

        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"

        input_img = content_img.clone()
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, num_steps=300)

        saveoutput = unloader(output[0])
        saveoutput.convert('RGB')

        output_path = "style_transfer/" + file
        saveoutput.save(output_path)
    time.sleep(0.05)

# print(temp)

# output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, input_img, num_steps=500)

# plt.figure()
# imshow(output, title='Output Image')
# print(output.size())
# saveoutput = unloader(output[0])
# saveoutput.convert('RGB')
# saveoutput.save("output/test.png")

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
