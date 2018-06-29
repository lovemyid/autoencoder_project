import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import  MNIST
from torchvision.utils import save_image
import csv
import test

if not os.path.exists('./test_img'):
    os.mkdir('./test_img')
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0,1)
    x = x.view(x.size(0), 3, 256, 256)
    return x
Epochs = 1000
LR = 0.0001

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# data_set = MNIST('./mnist', transform=img_transform)
data_set = test.myImageFloder('./data',transform=img_transform)

train_data = DataLoader(dataset=data_set, batch_size=16, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(64, momentum=0.1),
                                       nn.LeakyReLU(0.2, inplace=True)),
        # 128 x 128
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True)),
        # 64 x 64
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True)),
        # 32*32
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True)),
        # 16*16
            nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True)),
        # 8*8
            nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True)),
        # 4*4
            nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        )
        self.decoder = nn.Sequential(
            nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                         nn.BatchNorm2d(512, momentum=0.1),
                                         nn.ReLU()),
        # 4 x 4
            nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(512, momentum=0.1),
                                     nn.ReLU()),
        # 8*8
            nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(512, momentum=0.1),
                                     nn.ReLU()),
        # 16*16
            nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(256, momentum=0.1),
                                     nn.ReLU()),
        # 32*32
            nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(128, momentum=0.1),
                                     nn.ReLU()),
        # 64*64
            nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(64, momentum=0.1),
                                     nn.ReLU()),
        # 128*128
            nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.Sigmoid())
        )
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

ae = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=LR,
                             weight_decay=1e-5)
fileHeader =  ["epoch","loss"]
csvFile = open("loss.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(fileHeader)
for epoch in range(Epochs):
    for im in train_data:
        if torch.cuda.is_available():
            im = im.cuda()
        im = Variable(im)
        # 前向传播
        _, output = ae(im)
        loss = criterion(output, im)
        #im.shape[0]
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, Epochs, loss.data[0]/len(im)))
    d1 = [epoch, loss.data[0]/len(im)]
    writer.writerow(d1)
    # if epoch % 5 == 0:
    # pic = to_img(output.cpu().data)
    # save_image(pic, './test_img/image_{}.png'.format(epoch))
csvFile.close()



