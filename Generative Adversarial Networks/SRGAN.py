import os
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchsummary import summary
from torch import optim
from PIL import Image
import cv2
import glob
import numpy as np
from IPython.display import clear_output

!wget -O train_images_HR.zip http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
!wget -O test_images_LR.zip http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_difficult.zip
clear_output()

!unzip /content/train_images_HR.zip
!unzip /content/test_images_LR.zip
clear_output()

!rm /content/train_images_HR.zip
!rm /content/test_images_LR.zip
clear_output()

!mv /content/DIV2K_train_HR /content/train_images_HR
!mv /content/DIV2K_valid_LR_difficult /content/test_images_LR
clear_output()

class SRGAN_dataset(Dataset):
    def __init__(self, root_dir):
        super(SRGAN_dataset, self).__init__()
        images_path = os.path.join(root_dir, "*.png")
        self.data = glob.glob(images_path)
        self.high_res_size: int = 96
        self.low_res_size: int = int(self.high_res_size / 4)
        
    def common_transforms(self, image):
        transform = transforms.Compose([
            transforms.RandomCrop(size = self.high_res_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((90, 90)),
        ])
        image_t = transform(image)

        return image_t
    
    def high_res_transforms(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        image_t = transform(image)

        return image_t
    
    def low_res_transforms(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.low_res_size, self.low_res_size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        image_t = transform(image)

        return image_t
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert("RGB")
        image = self.common_transforms(image)
        image_hr = self.high_res_transforms(image)
        image_lr = self.low_res_transforms(image)

        return image_hr, image_lr

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True, activation=True, discriminator=False):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels) 
        if discriminator:
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.act = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm:
            out = self.bn(out)
        if self.activation:
            out = self.act(out)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=True)
        self.ps = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convblock0 = ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, batch_norm=True, activation=True, discriminator=False)
        self.convblock1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, batch_norm=True, activation=False, discriminator=False)

    def forward(self, x):
        out0 = self.convblock1(self.convblock0(x))
        out1 = out0 + x

        return out1

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, scale_factor):
        super().__init__()
        self.conv0 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=9, stride=1, padding=4, batch_norm=False, activation=True, discriminator=False)
        self.conv1 = nn.Sequential(*[ResidualBlock(in_channels=out_channels) for i in range(num_res_blocks)])
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True, activation=False, discriminator=False)
        self.up3 = UpsampleBlock(in_channels=out_channels, scale_factor=scale_factor)
        self.up4 = UpsampleBlock(in_channels=out_channels, scale_factor=scale_factor)
        self.conv5 = ConvBlock(in_channels=out_channels, out_channels=in_channels, kernel_size=9, stride=1, padding=4, batch_norm=False, activation=False, discriminator=False)

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out2 = out2 + out0
        out3 = self.up3(out2)
        out4 = self.up4(out3)
        out = self.conv5(out4)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv0 = ConvBlock(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, batch_norm=False, activation=True, discriminator=True)
        disc_blocks = []
        for i in range(num_blocks):
            self.in_channels = self.out_channels
            if(i%2):
                stride = 1
                self.out_channels *= 2
            else:
                stride = 2
            disc_blocks.append(ConvBlock(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     stride=stride,
                                     batch_norm=True, 
                                     activation=False, 
                                     discriminator=True
                                     )
                              )
        self.conv1 = nn.Sequential(*disc_blocks)
        self.fc2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(in_features=self.out_channels * 8 * 8, out_features=1024, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc2(self.conv1(self.conv0(x)))

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval()
        self.loss = nn.MSELoss()

        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
    
    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.loss(sr_features, hr_features)

low_resolution = 24  # 96x96 -> 24x24
x = torch.randn((1, 3, low_resolution, low_resolution))
gen = Generator(3, 64, 5, 2)
gen_out = gen(x)
disc = Discriminator(3, 64, 5)
disc_out = disc(gen_out)

print(gen_out.shape)
print(disc_out.shape)
print(disc_out)

transform = transforms.ToPILImage()
img = transform(gen_out[0])
img

#train discriminator
def discriminator_trainer(generator, discriminator, discriminator_loss, discriminator_optimizer, disc_epochs):
    generator.eval()
    discriminator.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(disc_epochs):
        for index, (high_res, low_res) in enumerate(train_data):
            high_res = high_res.to(device)
            low_res = low_res.to(device)

            super_res = generator(low_res)

            disc_sr = discriminator(super_res)
            disc_hr = discriminator(high_res)

            disc_loss_hr = discriminator_loss(disc_hr, torch.ones_like(disc_hr))
            disc_loss_sr = discriminator_loss(disc_sr, torch.zeros_like(disc_sr))

            disc_loss = disc_loss_hr + disc_loss_sr

            disc_loss.backward()
            discriminator_optimizer.step()

#train generator
def generator_trainer(generator, discriminator, content_loss, adversarial_loss, generator_optimizer):
    discriminator.eval()
    generator.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for index, (high_res, low_res) in enumerate(train_data):
        high_res = high_res.to(device)
        low_res = low_res.to(device)

        super_res = generator(low_res)
        disc_sr = discriminator(super_res)

        vgg_loss = content_loss(super_res, high_res)
        disc_loss = 1e-3 * adversarial_loss(disc_sr, torch.zeros_like(disc_sr))

        generator_loss = vgg_loss + disc_loss

        generator_loss.backward()
        generator_optimizer.step()

device = "cuda" if torch.cuda.is_available() else "cpu"

in_channels = 3
out_channels = 64
num_res_blocks = 16
scale_factor = 2
num_blocks = 7
epochs = 50
disc_epochs = 1
learning_rate = 1e-4
batch_size = 32
num_workers = 2

root_dir_train = "/content/train_images_HR"
train_dataset = SRGAN_dataset(root_dir_train)
train_data = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

root_dir_test = "content/test_images_HR"
test_dataset = SRGAN_dataset(root_dir_test)
test_data = DataLoader(test_dataset, batch_size=32, num_workers=2)

generator = Generator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_res_blocks=num_res_blocks,
                    scale_factor=scale_factor
                    ).to(device)

discriminator = Discriminator(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            num_blocks=num_blocks
                            ).to(device)

discriminator_loss = nn.BCEWithLogitsLoss()
content_loss = VGGLoss().to(device)
adversarial_loss = nn.BCEWithLogitsLoss()

discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print("Epoch {} started".format(epoch+1))
    discriminator_trainer(generator, discriminator, discriminator_loss, discriminator_optimizer, disc_epochs)
    generator_trainer(generator, discriminator, content_loss, adversarial_loss, generator_optimizer)
    print("Epoch {} completed".format(epoch+1))

def test_transforms(image):
        transform = transforms.Compose([
                                        transforms.RandomCrop(size = 24),                  
                                        transforms.ToTensor(),
                                    ])
        image_t = transform(image)
        return image_t

#a small testing script

root_dir = "/content/test_images_LR"
dataset = SRGAN_dataset(root_dir)
loader = DataLoader(dataset, batch_size=1, num_workers=0)
image = None
for high_res, low_res in loader:
    image = high_res
    break
pil_transform = transforms.ToPILImage()
lr_image = image
lr_image = pil_transform(lr_image[0])
lr_image

image = image.to(device)
sr_image = generator(image)

sr_image = pil_transform(sr_image[0])
sr_image