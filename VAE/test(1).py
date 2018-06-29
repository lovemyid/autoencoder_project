
import os
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transform
def default_loader(path):
    return Image.open(path).convert('RGB')


imag_transform =transform.Compose(
    [transform.Resize(256),
    transform.ToTensor()]
)
class myImageFloder(data.Dataset):
    def __init__(self, root, transform = None,loader=default_loader):
        self.transform = transform
        self.loader = loader
        imgs=[]
        class_name = []
        for filename in os.listdir('./data'):
            img = self.loader('./data'+'/'+filename)
            imgs.append(img)
            class_name.append(filename)
        self.root = root
        self.imgs = imgs
        self.classes = class_name


    def __getitem__(self, index):
        fn = self.classes[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
    def getName(self):
        return self.classes



# def testmyImageFloder():
#     dataloader = myImageFloder('./data',transform)
#     # print('dataloader.getName', dataloader.getName())
#
#     for index, img in enumerate(dataloader):
#         # plt.imshow(img)
#         # plt.show()
#         print(img.size)




if __name__ == "__main__":


    dataloader = myImageFloder('./data',transform=imag_transform)
    for img in dataloader:
        print(img.size())
    train_loader = torch.utils.data.DataLoader(dataloader, batch_size=2, shuffle=True)
    print(len(train_loader.dataset))