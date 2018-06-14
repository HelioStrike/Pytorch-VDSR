import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import *
from model import *

def main():
    os.makedirs('./images/', exist_ok=True)
    os.makedirs('./models/', exist_ok=True)

    #init weights
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if(classname.find('Conv') != -1 and classname.find('2d') != -1):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif (classname.find('Norm2d') != -1):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    #hyper-parameters
    num_epochs = 10
    lr = 0.1
    b1 = 0.5
    b2 = 0.999
    clip = 0.4

    #model save path
    model_save_path = "./models/srnet1.pth"

    #train data-loader
    train_loader = DataLoader(ImageDataset("../data/BSD100_SR", inp_folder="image_SRF_4", target_folder="image_SRF_2"),
                            batch_size=1, shuffle=True, num_workers=4)

    #loss function
    criterion = nn.MSELoss()

    #Super resolution network
    model = SRNet()
    if(torch.cuda.is_available()):
        model = model.cuda()
        criterion.cuda()
    model.apply(weights_init_normal)

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    #adjust learning rate
    def adjust_learning_rate():
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.8


    #see how well the model is doing by saving images
    def sample_images(epoch_num):
        imgs = next(iter(train_loader))
        inp = imgs['input'].cuda()
        target = imgs['target'].cuda()
        outs = model(inp)
        img_sample = torch.cat((inp.data, outs.data, target), 0)
        save_image(img_sample, './images/%s.png' % (epoch_num), nrow=3, normalize=True)

    print("Training...")
    #training the model
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            inps = batch['input'].cuda()
            targets = batch['target'].cuda()

            outs = model(inps)
            loss = criterion(outs, targets)

            epoch_loss += loss.data[0]

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()

            if(i%400==0):
                print("Analyzed", i+1, "images")
        print("Epoch", epoch, "loss:", epoch_loss)
        sample_images(epoch)
        adjust_learning_rate()
        torch.save(model.state_dict(), model_save_path)

if __name__=='__main__':
    main()
