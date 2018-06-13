import glob
import os
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, inp_folder="", target_folder="", tfms=None):
        self.inp_files = sorted(list(glob.glob(os.path.join(root, inp_folder) + "/*.*")))
        self.target_files = sorted(list(glob.glob(os.path.join(root, target_folder) + "/*.*")))

    def __getitem__(self, index):
        inp = Image.open(self.inp_files[index%len(self.inp_files)])
        target = Image.open(self.target_files[index%len(self.target_files)])

        tfms = transforms.Compose([transforms.ToTensor()])

        if(inp.size != target.size):
            inp_tfm = transforms.Compose([transforms.Resize((target.size[1], target.size[0])), transforms.ToTensor()])
            inp = inp_tfm(inp)
        else:
            inp = tfms(inp)
        target = tfms(target)

        return {'input': inp, 'target': target}

    def __len__(self):
            return len(self.inp_files)
