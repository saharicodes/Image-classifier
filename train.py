import argparse
import mymodel
import utils



if __name__ == '__main__':

    #argparse constructor
    parser = argparse.ArgumentParser(description='Train image classifier model')

    parser.add_argument('data_dir',
                         type=str, 
                         help='directorry of the image dataset')

    parser.add_argument('--save_dir', 
                        action='store', 
                        type=str,
                        nargs='?',
                        default='myapp_checkpoint.pth',
                        help='save directory of the model checkpoint'
                        )

    parser.add_argument('--arch', 
                        action='store', 
                        type=str,
                        nargs='?',
                        default="vgg16",
                        choices= ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                        help='model architecture'
                        )

    parser.add_argument('--learning_rate', 
                        action='store', 
                        type=float,
                        nargs='?',
                        default=0.001,
                        help='learning rate'
                        )

    parser.add_argument('--hidden_units', 
                        action='store', 
                        nargs='*',
                        default=[1024, 512],
                        type=int,
                        help='size of hidden layers'
                        )


    parser.add_argument('--epochs', 
                        action='store', 
                        type=int,
                        nargs='?',
                        default=5,
                        help='number of epochs for training'
                        )

    parser.add_argument('--gpu', 
                        action='store_true', 
                        help='enable gpu'
                        )

    myargs = parser.parse_args()  
    
    dataloaders, image_datasets = utils.load_prep_data(myargs.data_dir)
    
    model = mymodel.mymodel(arch=myargs.arch, hidden_units=myargs.hidden_units)
    mymodel.train_model(model, myargs.arch, dataloaders, image_datasets,
                        hidden_units=myargs.hidden_units, learning_rate=myargs.learning_rate, 
                        epochs=myargs.epochs, save_directory=myargs.save_dir, device=myargs.gpu)
    