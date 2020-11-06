import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps
import io
import base64
import sys
sys.path.append('..')

#1: CNN that was trained for our app
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1),
            torch.nn.ReLU(),
             torch.nn.Conv2d(32, 64, kernel_size=5, stride=1),
            torch.nn.ReLU()
        )
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(64*16*16, 10)
        )
    def img_transform(self, x):
        if list(x.size()[-2:]) == [28, 28]: # if image is already in adequate form
            return x
        trans = transforms.ToPILImage()
        revert_trans = transforms.ToTensor()
        container = []
        if len(x.size()) == 5: x = 255 - x[:, :, 3] # if RGBA, convert to grayscale
        for img in x:
            new_img = trans(img).resize((28, 28))
            new_img = ImageOps.grayscale(new_img) # if RGB converts to Grayscale
            container.append(revert_trans(new_img).unsqueeze(0)) # add extra dimension to make it valid argument
        x_new = torch.cat(container, 0)
        return x_new
    def linear_probs(self, vals):
        min_val = torch.min(vals)
        return torch.true_divide((vals - min_val), torch.sum(vals - min_val)) # gives probs dist between 0 and 1
    def forward(self, x):
        x = self.img_transform(x) # transforms images to 28x28
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = self.linear_layers(x)
        out = F.softmax(x, dim=1)
        return out, self.linear_probs(x)  # transform x to represent confidence scores (basically normalization), show in bar plot
    def __call__(self, x):
        x = self.img_transform(x) # transforms images to 28x28
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = self.linear_layers(x)
        x = F.softmax(x, dim=1)
        return np.argmax(x.detach().numpy()) 

# 2: Convert base64-encoded string into numpy array
def string2img(string):
    string = string.split('base64,')[1]
    decoded = base64.b64decode(string)
    buffer = io.BytesIO(decoded)
    im = Image.open(buffer)
    return np.asarray(im)

# 3: Take image path and return base64 encoded image
def img2string(image_path):
    with open(image_path, "rb") as img_file:
        return 'data:' + image_path + ';base64,' + base64.b64encode(img_file.read()).decode('utf-8')