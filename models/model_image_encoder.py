import sys

sys.path.append('/home/mlspeech/gshalev/gal/IC_NLI')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
import torch
import torchvision
from torch import nn


class Image_encoder(nn.Module):

    def __init__(self, run_device, out_dim):
        super(Image_encoder, self).__init__()

        self.run_device = run_device
        self.out_dim = out_dim
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.fc = nn.Linear(1000, out_dim)


    def forward(self, img):
        out = self.resnet(img)
        out = torch.relu(out)
        out = self.fc(out)
        return out

# model_image_encoder.py