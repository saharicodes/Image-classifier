import torch
from torch import nn, optim
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np
from utils import process_image
import pathlib

def mymodel(*, arch="vgg16", hidden_units=[1024, 512]):
    """Builds a vgg network with arbitrary architecture and size of hidden layers

    Keyword Arguments:
        arch {str} -- nn network architecture (default: {"vgg16"})
        hidden_units {list} -- size of hidden layers (default: {[1024, 512]})
    """   

    #Load pretrained vgg model
    model = eval("models." + arch + "(pretrained=True)")

    #Freeze model parameters
    for p in model.parameters():
        p.requires_grad = False 

    #Build the new classifier
    layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
    linearunits =  [nn.Linear(25088, hidden_units[0])]   
    linearunits.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])   
    layers = []
    for h in linearunits:
        layers.extend([h, nn.ReLU(), nn.Dropout(p=0.2)])

    layers.extend([nn.Linear(hidden_units[-1], 102), nn.LogSoftmax(dim=1)])

    classifier = nn.Sequential(*layers)

    model.classifier = classifier

    return model


def train_model(model, arch, dataloaders, image_datasets, *,
                hidden_units=[1024, 512], learning_rate=0.001, 
                epochs=5, save_directory="myapp_checkpoint.pth", device=False ):
  
    """Train image classifier network
    Arguments:
        model {object} -- image classifier cnn model
        dataloaders {generator} -- generator to load batch of transformed images


    Keyword Arguments:
        learning_rate {float} -- learning rate (default: {0.001})
        epochs {int} -- number of epochs for training (default: {5})
        save_directory {str} -- save directory (default: {"mycheckpoint.pth"})
        device {str} -- device to be used (default: {"cpu"})
    """
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    #Train the new classifier
    torch.cuda.empty_cache()
    if device:
        device = 'cuda'
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)
    
    running_loss = 0
    training_losses = []
    validation_losses = []

    for e in range(epochs):
        
        for images, labels in dataloaders["train"]:
            images = images.to(device)
            labels = labels.to(device)                
            optimizer.zero_grad()          
            logps = model.forward(images)        
            loss = criterion(logps, labels)        
            loss.backward()        
            optimizer.step()                
            running_loss += loss.item()   
            
        else:
            validation_loss = 0 
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in dataloaders["validation"]:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    validation_loss += batch_loss.item()
                    
                    #Calculate accuracy during validation  
                    ps = torch.exp(logps)
                    top_val, top_class = ps.topk(1, dim =1)
                    equals = top_class == labels.view( *top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
            print(f"At epoch {e+1}/{epochs}, training loss: {running_loss/len(dataloaders['train']):.3f}, "\
                  f"validation loss: {validation_loss/ len(dataloaders['validation']):.3f}, "\
                  f"accuracy: {accuracy/len(dataloaders['validation']):.3f}")
            
            model.train()
            training_losses.append(running_loss)
            validation_losses.append(validation_loss)
            running_loss = 0

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'epochs': epochs,
                  'arch': arch,
                  'training_losses': training_losses,
                  'validation_losses': validation_losses, 
                  'class_to_idx': model.class_to_idx, 
                  'input': 25088,
                  'output': 102,        
                  'hidden_layers': hidden_units,          
                  'optimizer_state_dict': optimizer.state_dict(),          
                  'model_state_dict': model.state_dict()}

    torch.save(checkpoint, save_directory)


def reload_model(save_directory="myapp_checkpoint.pth"):
    
    checkpoint = torch.load(save_directory)
    model = mymodel(arch=checkpoint["arch"], hidden_units=checkpoint["hidden_layers"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def predict(image_path, model, topk=1, device=False, cat_to_name=None ):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    img_cat = pathlib.Path(image_path).parent.name
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    
    if device:
        model.to('cuda')
        img = img.to('cuda')
    else:
        model.to('cpu')
        img = img.to('cpu')

    model.eval()
    with torch.no_grad():
        logps = model.forward(img)
    ps = torch.exp(logps)
    probs, idxs = ps.topk(topk)
    idxs = idxs.cpu().numpy()[0]
    probs = probs.cpu().numpy()[0]
    idxs = idxs.tolist()
    probs = probs.tolist()
    #revert class to idx map
    idx_to_class = {val: key for key, val in model.class_to_idx.items()} 
    classes = [idx_to_class[idx] for idx in idxs]

    if cat_to_name:
        flower_names = [cat_to_name[c] for c in classes]
    else:
        flower_names = None


    return probs, classes, flower_names