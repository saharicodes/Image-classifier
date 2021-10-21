import torch
from torchvision import transforms, datasets
import pathlib
from PIL import Image
import numpy as np
import json


def load_prep_data(data_directory):
    """Load and preprocess image data

    Arguments:
        data_directory {str} -- [directory of the image dataset]
    """    
    # set data_dirs
    data_dir = pathlib.Path(data_directory).resolve()
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    test_dir = data_dir / 'test'

    #Define your transforms for the training, validation, and testing sets
    data_transforms = {"train": transforms.Compose([transforms.RandomRotation(30), 
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                        
                        "validation": transforms.Compose([transforms.Resize(255), 
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                    
                        "test": transforms.Compose([transforms.Resize(255), 
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                        }
                   

    #Load the datasets with ImageFolder
    image_datasets = {"train": datasets.ImageFolder(train_dir, transform = data_transforms["train"]),
                      "validation": datasets.ImageFolder(valid_dir, transform = data_transforms["validation"]), 
                      "test": datasets.ImageFolder(test_dir, transform = data_transforms["test"] )
                     }

    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size = 16, shuffle = True),
                   "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size = 16), 
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size = 16)
                  }   

    
    return dataloaders, image_datasets

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    # load image
    img = Image.open(image)
    size_img = img.size
    
    # resize image
    if size_img[0] > size_img[1]:
        img.thumbnail((50000, 256))
    else:
        img.thumbnail((256, 50000))
    
    # crop image
    size_img = img.size
    img = img.crop((size_img[0]//2 -(224/2),
                    size_img[1]//2 - (224/2),
                    size_img[0]//2 +(224/2),
                    size_img[1]//2 + (224/2) 
                    ))
    # normalize image color
    img = np.array(img)/255
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # adjust np array dimension
    img = img.transpose((2, 0, 1))
    
    return img


def load_cat_name(jason_path='cat_to_name.json'):
    with open(jason_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name