import torch
import glob
import argparse, os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import *

#arg parser
parser = argparse.ArgumentParser(description="Apply VDSR")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./models/srnet1.pth", type=str, help="model path")
parser.add_argument("--imagepath", default="./testimages", type=str, help="image path")
parser.add_argument("--savepath", default="./saveimages", type=str, help="image save path")
parser.add_argument("--height", default=0, help="output height")
parser.add_argument("--width", default=0, help="output width")
opt = parser.parse_args()

cuda = opt.cuda
os.makedirs(opt.savepath, exist_ok=True)

model = SRNet()
model.load_state_dict(torch.load(opt.model))

if(cuda):
    model = model.cuda()

#transforms to be applied on each image before entering the model
if(opt.height == 0 or opt.width == 0):
    tfms = transforms.Compose([transforms.ToTensor()])
else:
    tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize(opt.height, opt.width)])

#we pass each image in imagepath through the model and save it in savepath
for file in list(glob.glob(opt.imagepath + '/*.*')):
    img = tfms(Image.open(file)).unsqueeze(0)
    if(cuda):
        img = img.cuda()

    out = model(img)

    print(file)
    save_image(out, os.path.join(opt.savepath, file.split('\\')[-1]))
