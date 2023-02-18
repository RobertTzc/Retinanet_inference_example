import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_dir', type = str,
                        help =' the directory of the model,default using the Bird_drone_KNN model included',
                        default='./checkpoint/Bird_drone_KNN/final_model.pkl')
    parser.add_argument('--model_type', type = str,
                        help =' the type of the model,default type used is Bird_drone_KNN',
                        default='Bird_drone_KNN')
    parser.add_argument('--image_root',type = str,
                        help = 'The root dir where image are stores')
    parser.add_argument('--image_ext',type = str, default = 'JPG',
                        help = 'the extension of the image(without dot), default is JPG')
    parser.add_argument('--image_altitude',type = int, default = 15,
                        help = 'the altitude of the taken image, default is set to be 15')
    parser.add_argument('--image_location',type = str, default = 'No_Where',
                        help = 'the location of the taken image, default is set to be No_Where')
    parser.add_argument('--image_date',type = str, default = '2022-10-26',
                        help = 'the date of the taken image, default is set to be 2022-10-26')
    parser.add_argument('--use_altitude',type = bool, default = True,
                        help = 'whether to use altitude to scale the image, default is True')
    parser.add_argument('--out_dir',type = str,
                        help = 'where the output will be generated,default is ./results',
                        default = './results')
    parser.add_argument('--visualize',type = bool,
                        help = 'whether to have visualization stored to result, default is True',
                        default = True)
    parser.add_argument('--evaluate',type = bool,
                        help = 'whether to evaluate the reslt,default is False',
                        default = False)
    args = parser.parse_args()
    
    #if the image_root input is with extension(*.JPG) wrap into list
    #else fetch the list of image
    return args