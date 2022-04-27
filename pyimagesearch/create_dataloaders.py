# import the necessary packages
from . import config
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import os


"""Custom dataset that includes extends torchvision.datasets.ImageFolder
    to override getitem to return image filename in tuple.
"""
class ImageFolderWithPaths(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(ImageFolderWithPaths, self).__init__(root=root, transform=transform)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_dataloader(rootDir, transforms, batchSize, shuffle=True):
    # create a dataset and use it to create a data loader
    ds = ImageFolderWithPaths(root=rootDir, transform=transforms)
    loader = DataLoader(ds, batch_size=batchSize,
        shuffle=shuffle,
        num_workers=os.cpu_count(),
        pin_memory=True if config.DEVICE == "cuda" else False)

    # return a tuple of  the dataset and the data loader
    return (ds, loader)

