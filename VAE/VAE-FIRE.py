import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
import numpy as np
import os
import test

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')


def to_img(x):
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)
    return x


num_epochs = 1000
# batch_size = 50
learning_rate = 1e-4

img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset = MNIST('./mnist', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_set = test.myImageFloder('./data',transform=img_transform)

train_data = DataLoader(dataset=data_set, batch_size=16, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        # 256 x 256
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 32*32
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 16*16
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 8*8
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 4*4
        self.conv7 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512, momentum=0.1),
                                   nn.LeakyReLU(0.2, inplace=True))
        # 2*2
        self.fc11 = nn.Linear(512*2*2, 512*2*2)
        self.fc12 = nn.Linear(512*2*2, 512*2*2)
        self.fc2 = nn.Linear(512*2*2, 512*2*2)


        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(512, momentum=0.1),
                                     nn.ReLU())
        # 4 x 4
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(512, momentum=0.1),
                                     nn.ReLU())
        # 8*8
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(512, momentum=0.1),
                                     nn.ReLU())
        # 16*16
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(256, momentum=0.1),
                                     nn.ReLU())
        # 32*32
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(128, momentum=0.1),
                                     nn.ReLU())
        # 64*64
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(64, momentum=0.1),
                                     nn.ReLU())
        # 128*128
        self.deconv7 = nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.Sigmoid())


    def encode(self, x):
        # input: noise output: mu and sigma

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return self.fc11(out.view(out.size(0), -1)), self.fc12(out.view(out.size(0), -1))
        #平均值，方差对数

    def reparametrize(self, mu, logvar):
        var = logvar.mul(0.5).exp_()  #标准差
        eps = torch.FloatTensor(var.size()).normal_()
        #标准正态分布
        eps = Variable(eps)
        out = eps.mul(var).add_(mu)
        return out
    def decode(self, z):
        out = self.fc2(z)
        out = self.deconv1(out.view(z.size(0), 512, 2, 2))
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        out = self.deconv7(out)
        return out


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


model = VAE()
reconstruction_function = nn.MSELoss(size_average=False)


model.load_state_dict(torch.load('./model/net_params_446.pkl'))
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    CEL = reconstruction_function(recon_x, x)  # bce loss
    # CEL = F.binary_cross_entropy(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return CEL, KLD


optimizer = optim.Adam(model.parameters(), lr=learning_rate,)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# data = torch.Tensor(2, 3 * 256 * 256)
# data = Variable(data)
# def adjust_learning_rate(optimizer, epoch):
#     """
#     每50个epoch,学习率以0.9的速率衰减
#     """
#     lr = learning_rate * (0.9 ** (epoch // 50))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer





for epoch in range(num_epochs):
    model.train()
    # train_loss = 0
    # optimizer_decay = adjust_learning_rate(optimizer, epoch)
    for batch_idx, images in enumerate(train_data):
        # data.data.resize_(images.size()).copy_(images)
        # if torch.cuda.is_available():
        #     img = img.cuda()
        images_v = Variable(images)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(images_v)
        CEL, KLD = loss_function(recon_batch, images_v, mu, logvar)
        loss = CEL+KLD
        loss.backward()
        # train_loss += loss.data[0]
        optimizer.step()
        # scheduler.step()
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch,
        #         batch_idx * len(data),
        #         len(dataloader.dataset), 100. * batch_idx / len(dataloader),
        #         loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f} CELoss: {:.4f} KLD: {:.4f}'.format(
        epoch, loss.data[0]/len(images_v), CEL.data[0]/len(images_v), KLD.data[0]/len(images_v)))
    # if epoch % 10 == 0:
    save = to_img(recon_batch.cpu().data)
    save_image(save, './vae_img/re1/image_{}.png'.format(epoch+447))
    torch.save(model.state_dict(), './model/net_params_{}.pkl'.format(epoch+447))
