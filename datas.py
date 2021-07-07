import torch
import torchvision
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision import utils as vutils
from torch.utils.data import Dataset, DataLoader
import os

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


class Img_Dataset(Dataset):
    def __init__(self, img_path, cuda_device, transform=None):
        self.cuda_device = cuda_device
        self.img_path = img_path
        self.transform = transform
        self.img_list = sorted(os.listdir(img_path))

    def __getitem__(self, index):
        img_name = self.img_path + str(index) + '.jpg'
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img).to(self.cuda_device)
        return img

    def __len__(self):
        return len(self.img_list)

def data_split(full_dataset, split_num=None, split_ratio=None):
    data_size = len(full_dataset)
    train_size = data_size
    valid_size = 0
    test_size = 0

    if split_num is not None:
        train_size = split_num[0]
        valid_size = split_num[1]
        test_size = split_num[2]
    elif split_ratio is not None:
        train_size = int(data_size * split_ratio[0])
        valid_size = int(data_size * split_ratio[1])
        test_size = data_size - train_size - valid_size
    
    train_indices = list(range(0, train_size))
    valid_indices = list(range(train_size, train_size + valid_size))
    test_indices = list(range(train_size + valid_size, data_size))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SequentialSampler(valid_indices)
    test_sampler = torch.utils.data.SequentialSampler(test_indices)

    return train_sampler, valid_sampler, test_sampler


def CelebA(batch_size, device):
    transform = transforms.Compose([transforms.Resize((128, 128), Image.ANTIALIAS),
                                    transforms.ToTensor()])
    
    full_dataset = Img_Dataset(img_path='/home/vision/diska3/Data/CelebA-HQ-img/', cuda_device=device, transform=transform)
    train_sampler, valid_sampler, test_sampler = data_split(full_dataset, split_ratio=[0.8, 0.2, 0.0])
    train_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

if __name__ == '__main__':
    train_loader, valid_loader = CelebA(1, 0)
    for data in train_loader:
        img = data
        print(img.shape)
        save_image_tensor(img, '/home/vision/diska2/1JYK/RL/Project/1.jpg')
        break
