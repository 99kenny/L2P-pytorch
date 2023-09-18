from torchvision import transforms, datasets
import torch
from urllib.request import urlretrieve
import zipfile
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
# dataset name, transform_train, transform_val, data_path
def get_dataset(dataset, transform_train, transform_val, data_path, download):
    if dataset == "CIFAR10":
        train = datasets.CIFAR10(data_path, train=True, download=download, transform=transform_train)
        val = datasets.CIFAR10(data_path, train=False, download=download, transform=transform_train)
    elif dataset == "CIFAR100":
        train = datasets.CIFAR100(data_path, train=True, download=download, transform=transform_train)
        val = datasets.CIFAR100(data_path, train=False, download=download, transform=transform_train)
    elif dataset == "MNIST":
        train = datasets.MNIST(data_path, train=True, download=download, transform=transform_train)
        val = datasets.MNIST(data_path, train=False, download=download, transform=transform_train)
    elif dataset == "Fashion-MNIST":
        train = datasets.FashionMNIST(data_path, train=True, download=download, transform=transform_train)
        val = datasets.FashionMNIST(data_path, train=False, download=download, transform=transform_train)
    elif dataset == "SVHN":
        train = datasets.SVHN(data_path, train=True, download=download, transform=transform_train)
        val = datasets.SVHN(data_path, train=False, download=download, transform=transform_train)
    elif dataset == "notMNIST":
        root = data_path
        if download:
            # data url
            data_url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip"
            zip_file_path = "{}/notMNIST.zip".format(root)
            # retrieve data 
            print("Downloading notMNIST from https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip")
            path, headers = urlretrieve(data_url, zip_file_path)
            # unzip
            with zipfile.ZipFile(zip_file_path, 'r') as obj:
                obj.extractall(root)
        
        train = datasets.ImageFolder("{}/notMNIST/Train".format(root), transform=transform_train)
        val = datasets.ImageFolder("{}/notMNIST/Train".format(root), transform=transform_val)
        
    else :
        raise ValueError("{} not found".format(dataset))
    return train, val

def get_transforms(is_train, *args):
    # train dataset transform
    if is_train:
        '''
        return transforms.Compose(
            args
        )
        '''
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=(224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
    # test dataset transform
    else :
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        
# check if the dataset is valid
def print_img(dataset, idx):
    img = dataset.__getitem__(idx)[0]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(to_pil_image(img), cmap='gray')
