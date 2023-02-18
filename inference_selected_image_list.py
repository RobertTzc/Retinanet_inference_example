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
import sys
import glob
import pandas as pd
from args import *
from pyexiv2 import Image
from utils import read_LatLotAlt,get_GSD
from WaterFowlTools.mAp import mAp_calculate,plot_f1_score,plot_mAp
import matplotlib.pyplot as plt
from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image
warnings.filterwarnings("ignore")
from inference_image_list import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_conf_threshold = {'Bird_A':0.2,'Bird_B':0.2,'Bird_C':0.2,'Bird_D':0.2,'Bird_E':0.2,'Bird_drone':0.2}
model_extension = {'Bird_drone':{40:('_alt_30',30),75:('_alt_60',60),90:('_alt_90',90)}}




if __name__ == '__main__':
    h = [0,20,40,70,120]
    for idx in range(4):
        if (idx==0):
            continue
        image_root = '/home/robert/Data/WaterFowl_Processed/drone_collection'
        df = pd.read_csv('/home/robert/Data/drone_collection/image_info_no_decoy.csv')
        df = df[(df['height']>=h[idx-1])&(df['height']<=h[idx])]
        image_list = [image_root+'/'+i for i in df['image_name']]
        altitude_list = [int(i) for i in list(df['height'])]
        location_list = ['test' for _ in image_list]
        date_list = ['2023' for _ in image_list]
        target_dir = './results/{}'.format(h[idx])
        model_type = 'Bird_drone_KNN'
        os.makedirs(target_dir,exist_ok=True)
        model_dir = '/home/robert/Models/Retinanet_inference_example/checkpoint/Bird_drone_KNN/final_model.pkl'
        image_out_dir = os.path.join(target_dir,'visualize-results')
        text_out_dir = os.path.join(target_dir,'detection-results')
        csv_out_dir = os.path.join(target_dir,'detection_summary.csv')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print ('*'*30)
        print ('Using model type: {}'.format(model_type))
        print ('Using device: {}'.format(device))
        print ('Image out dir: {}'.format(image_out_dir))
        print ('Texting out dir: {}'.format(text_out_dir))
        print ('Inferencing on Images:\n {}'.format(image_list))
        print ('Altitude of each image:\n {}'.format(altitude_list))
        print ('*'*30)
        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(text_out_dir, exist_ok=True)
        inference_mega_image_Retinanet(
            image_list=image_list, model_dir = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
            scaleByAltitude=True, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list,
            visualize = True,device = device,model_type = model_type)
  
        try:
            precision, recall, sum_AP, mrec, mprec, area = mAp_calculate(image_name_list = image_name_list, 
                                                                        gt_txt_list=[os.path.splitext(i)[0]+'.txt' for i in image_list],
                                                                        pred_txt_list = [text_out_dir+'/'+os.path.splitext(i)[0]+'.txt' for i in image_name_list],
                                                                        iou_thresh=0.3)
            plot_f1_score(precision, recall, args.model_type, text_out_dir, area, 'f1_score', color='r')
            plt.legend()
            plt.savefig(os.path.join(target_dir,'f1_score.jpg'))
            plt.figure()
            plot_mAp(precision, recall, mprec, mrec,  args.model_type, area, 'mAp', color='r')
            plt.legend()
            plt.savefig(os.path.join(target_dir,'mAp.jpg'))
            print('Evaluation completed, proceed to wrap result')
        except:
            print('Failed to evaluate, Skipped')

    
