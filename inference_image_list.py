import cv2
import torch
import torchvision.transforms as transforms
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont
import os
import warnings
import glob
import os
import numpy as np
from PIL import Image
import numpy as np
import time
import json
from pyexiv2 import Image
from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_GSD(altitude,camera_type='Pro2', ref_altitude=60):

    if (camera_type == 'Pro2'):
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * altitude)/(10.26*5472)
    elif (camera_type == 'Air2'):
        ref_GSD = (6.4*ref_altitude)/(4.3*8000)
        GSD = (6.4*altitude)/(4.3*8000)
    else:
        ref_GSD = (13.2 * ref_altitude)/(10.26*5472)
        GSD = (13.2 * altitude)/(10.26*5472)
    return GSD, ref_GSD


def inference_mega_image_Retinanet(image_list, model_dir, image_out_dir,text_out_dir, visualize,scaleByAltitude=True, defaultAltitude=[],**kwargs):
    if (kwargs['device']!=torch.device('cuda')):
        print ('loading CPU mode')
        device = torch.device('cpu')
        net = torch.load(model_dir,map_location=device)
        net = net.module.to(device)
    else:
        device = torch.device('cuda')
        net = torch.load(model_dir)
    net.to(kwargs['device'])
    print('check net mode',next(net.parameters()).device)
    encoder = DataEncoder(device)
    record = []
    for idxs, image_dir in (enumerate(image_list)):
        start_time = time.time()
        try:
            altitude = get_image_taking_conditions(image_dir)['altitude']
            print ('Processing image name: {} with Altitude of {}'.format(os.path.basename(image_dir),altitude))
        except:
            altitude = int(defaultAltitude[idxs])
            print ('Using default altitude for: {} with Altitude of {}'.format(os.path.basename(image_dir),altitude))
        if scaleByAltitude:
            GSD,ref_GSD = get_GSD(altitude,camera_type='Pro2', ref_altitude=90) # Mavic2 Pro GSD equations
            ratio = 1.0*ref_GSD/GSD
        else:
            ratio = 1.0
        print('Processing scale {}'.format(ratio))
        bbox_list = []
        mega_image = cv2.imread(image_dir)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_BGR2RGB)
        sub_image_list, coor_list = get_sub_image(
            mega_image, overlap=0.2, ratio=ratio)
        for index, sub_image in enumerate(sub_image_list):
            with torch.no_grad():
                inputs = transform(cv2.resize(
                    sub_image, (512, 512), interpolation=cv2.INTER_AREA))
                inputs = inputs.unsqueeze(0).to(kwargs['device'])
                loc_preds, cls_preds = net(inputs)
                boxes, labels, scores = encoder.decode(
                    loc_preds.data.squeeze(), cls_preds.data.squeeze(), 512, CLS_THRESH = 0.2,NMS_THRESH = 0.5)
            if (len(boxes.shape) != 1):
                for idx in range(boxes.shape[0]):
                    x1, y1, x2, y2 = list(
                        boxes[idx].cpu().numpy())  # (x1,y1, x2,y2)
                    score = scores.cpu().numpy()[idx]
                    bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1,
                                     coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2, score])
        txt_name = image_dir.split('/')[-1].split('.')[0]+'.txt'
        with open(os.path.join(text_out_dir,txt_name), 'w') as f:
            if (len(bbox_list) != 0):
                bbox_list = np.asarray([box for box in bbox_list])
                box_idx = py_cpu_nms(bbox_list, 0.25)
                num_bird = len(box_idx)
                selected_bbox = bbox_list[box_idx]
                print('Finished on {},\tfound {} birds'.format(
                os.path.basename(image_dir), len(selected_bbox)))
                selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                for box in selected_bbox:
                    f.writelines('bird {} {} {} {} {}\n'.format(
                        box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    if (visualize):#Only display conf Thresh>0.3 bbox
                        cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                            box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(mega_image, (int(box[0]), int(
                            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_RGB2BGR)
        if (visualize):
            cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
        record.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                       kwargs['lat_list'][idxs],kwargs['lot_list'][idxs],defaultAltitude[idxs],num_bird,time.time()-start_time])
    record = pd.DataFrame(record)
    record.to_csv(kwargs['csv_out_dir'],header = ['image_name','date','location','latitude','longitude','altitude','num_birds','time_cost'],index = True)
    
        

import argparse
import sys
import pandas as pd
def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_dir', type = str,
                        help =' the directory of the model',
                        default='./checkpoint/Bird_D/final_model.pkl')
    parser.add_argument('--model_type', type = str,
                        help =' the type of the model',
                        default='Bird_B')
    # parser.add_argument('--altitude_list',nargs='+',
    #                     help = 'altitude list of the input image')
    parser.add_argument('--csv_dir',type = str,
                         help = 'csv_file of the input list')
    parser.add_argument('--image_root',type = str,
                        help = 'The root dir where image are stores')
    parser.add_argument('--out_dir',type = str,
                        help = 'where the output will be generated',
                        default = './results')
    parser.add_argument('--visualize',type = bool,
                        help = 'whether to have visualize',
                        default = True)
    parser.add_argument('--evaluate',type = bool,
                        help = 'whether to evaluate',
                        default = False)
    parser.add_argument('--ext',type = str,default = 'JPG',
                    help = 'extension of the image name without . ',)
    args = parser.parse_args()
    
    #if the image_root input is with extension(*.JPG) wrap into list
    #else fetch the list of image
    return args

if __name__ == '__main__':
    args = get_args()
    model_type = args.model_type
    df = pd.read_csv(args.csv_dir)
    image_list = [os.path.join(args.image_root,i) for i in df['image_name']]
    print (image_list)

    altitude_list = df['altitude']
    location_list = df['location']
    lat_list = df['latitude']
    lot_list = df['longitude']
    date_list = df['date']
    
    print(altitude_list)
    # if (dataset_folder=='Bird_A'):
    #     altitude_list = [150,80,70,50] #Bird A height info stores inside the file names, here we just maunally input it
    # else:
    #     altitude_list = 90 #if meta data is not avaliable, we use 90 meters for all images
    target_dir = args.out_dir
    model_dir = args.model_dir
    image_out_dir = target_dir+'/visualize-results'
    text_out_dir = target_dir+'/detection-results'
    csv_out_dir = target_dir+'/detection_summary.csv'
    device = device
    print ('*'*20)
    print ('Using model type: {}'.format(model_type))
    print ('Using device: {}'.format(device))
    print ('Image out dir: {}'.format(image_out_dir))
    print ('Texting out dir: {}'.format(text_out_dir))
    print ('Inferencing on Images:\n {}'.format(image_list))
    print ('Altitude of each image:\n {}'.format(altitude_list))
    print ('Visualize on each image:\n {}'.format(args.visualize))
    print ('*'*20)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)
    # inference_mega_image_Retinanet(
	# 	image_list=image_list, model_dir = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
	# 	scaleByAltitude=True, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list,
    #     lat_list = lat_list,lot_list = lot_list,visualize = args.visualize,device = device)
    if (args.evaluate):
        from WaterFowlTools.mAp import mAp_calculate,plot_f1_score,plot_mAp
        import matplotlib.pyplot as plt
        precision, recall, sum_AP, mrec, mprec, area = mAp_calculate(image_name_list = (df['image_name']), 
                                                                     gt_txt_list=[os.path.splitext(i)[0]+'.txt' for i in image_list],
                                                                     pred_txt_list = [text_out_dir+'/'+os.path.splitext(i)[0]+'.txt' for i in (df['image_name'])],
                                                                     iou_thresh=0.3)
        plot_f1_score(precision, recall, args.model_type, text_out_dir, area, 'f1_score', color='r')
        plt.savefig(target_dir+'/f1_score.jpg')
        plt.figure()
        plot_mAp(precision, recall, mprec, mrec,  args.model_type, area, 'mAp', color='r')
        plt.savefig(target_dir+'/mAp.jpg')
    argparse_dict = vars(args)
    with open(target_dir+'/configs.json','w') as f:
        json.dump(argparse_dict,f,indent=4)