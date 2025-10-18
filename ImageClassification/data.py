import torch
from torchvision import datasets, transforms
import numpy as np




SHAPES = {
    "cifar10": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1)
}

attribute_indices = {
    'smiling': 31,
    'male': 20
}

class CelebADatasetProcess(torch.utils.data.Dataset):
    def __init__(self, celeba_dataset, attribute_name= None):
        self.celeba_dataset = celeba_dataset
        if attribute_name is not None:
            self.attribute_idx = [attribute_indices[attribute_name]]
        else: 
            self.attribute_idx = np.arange(40)

    def __len__(self):
        return len(self.celeba_dataset)

    def __getitem__(self, idx):
        image, attributes = self.celeba_dataset[idx]
        attribute_label = attributes[self.attribute_idx]
        return image, attribute_label.float()



def get_data(name, augment=False, **kwargs):
    if name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CIFAR10(root="data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root="data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ), download=True)

    elif name == "fmnist":
        train_set = datasets.FashionMNIST(root='data', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='data', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)

    elif name == "mnist":
        train_set = datasets.MNIST(root='data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
        
    elif name=='imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
        transform = [transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    ]
       
        train_set = datasets.ImageFolder(root='imagenette2-160/train', 
                                         transform=transforms.Compose(transform))
        test_set = datasets.ImageFolder(root='imagenette2-160/val', 
                                        transform=transforms.Compose(transform))
        

    elif name == 'multilabel-celeba':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

       
        comp_train_set = datasets.CelebA(root='CelebA', split='train', 
                                            transform=transform, download=False)
        comp_test_set = datasets.CelebA(root='CelebA', split='test', 
                                            transform=transform, download=False)
        
        train_set = CelebADatasetProcess(comp_train_set)
        test_set = CelebADatasetProcess(comp_test_set)


    elif name == 'male-celeba':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

       
        comp_train_set = datasets.CelebA(root='CelebA', split='train', 
                                            transform=transform, download=False)
        comp_test_set = datasets.CelebA(root='CelebA', split='test', 
                                            transform=transform, download=False)

        # Create datasets for the "Male" attribute
        train_set = CelebADatasetProcess(comp_train_set, 'male')
        test_set = CelebADatasetProcess(comp_test_set, 'male')

    elif name == 'smiling-celeba':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

       
        comp_train_set = datasets.CelebA(root='CelebA', split='train', 
                                            transform=transform, download=False)
        comp_test_set = datasets.CelebA(root='CelebA', split='test', 
                                            transform=transform, download=False)

        # Create datasets for the "Smiling" attribute
        train_set = CelebADatasetProcess(comp_train_set, 'smiling')
        test_set = CelebADatasetProcess(comp_test_set, 'smiling')

        
    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set


