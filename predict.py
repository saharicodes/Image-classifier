import argparse
import mymodel
import utils



if __name__ == '__main__':
    #argparse constructor
    parser = argparse.ArgumentParser(description='predict image class')

    parser.add_argument('input',
                         type=str, 
                         help='path to image')

    parser.add_argument('checkpoint',
                         type=str, 
                         help='path to checkpoint')

    parser.add_argument('--top_k', 
                        action='store', 
                        type=int,
                        nargs='?',
                        default=1,
                        help='top categories'
                        )

    parser.add_argument('--category_names', 
                        action='store', 
                        type=str,
                        nargs='?',
                        default='cat_to_name.json',
                        help='path to cat_to_name.json'
                        )

    
    parser.add_argument('--gpu', 
                        action='store_true', 
                        help='enable gpu'
                        )


    myargs = parser.parse_args()                    

    cat_to_name = utils.load_cat_name(jason_path=myargs.category_names)

    model = mymodel.reload_model(save_directory=myargs.checkpoint)

    probs, classes, flower_names = mymodel.predict(myargs.input, model, topk=myargs.top_k, device=myargs.gpu, cat_to_name=cat_to_name )
    print(f"top probabilities: {probs} ",
          f"top categories: {classes}",
          f"top flower names: {flower_names}")