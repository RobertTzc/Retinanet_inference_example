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

from pyexiv2 import Image
from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

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


def inference_mega_image_Retinanet(image_list, model_dir, target_dir, scaleByAltitude=True, defaultAltitude=90):
    net = torch.load(model_dir)
    net.cuda()
    encoder = DataEncoder()
    if (not isinstance(defaultAltitude,list)):
        defaultAltitude = len(image_list)*[defaultAltitude]
    for idxs, image_dir in (enumerate(image_list)):
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
                inputs = inputs.unsqueeze(0).cuda()
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
        print('Finished on {},\tfound {} birds'.format(
            os.path.basename(image_dir), len(bbox_list)))
        txt_name = image_dir.split('/')[-1].split('.')[0]+'.txt'
        with open(text_out_dir+'/'+txt_name, 'w') as f:
            if (len(bbox_list) != 0):
                bbox_list = np.asarray([box for box in bbox_list])
                box_idx = py_cpu_nms(bbox_list, 0.25)
                for box in bbox_list[box_idx]:
                    f.writelines('bird {} {} {} {} {}\n'.format(
                        box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    if (box[4]>=0.3):#Only display conf Thresh>0.3 bbox
                        cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                            box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(mega_image, (int(box[0]), int(
                            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_out_dir+'/'+image_dir.split('/')[-1], mega_image)


if __name__ == '__main__':
    dataset_folder = 'Bird_B'
    if (dataset_folder=='Bird_A'):
        altitude_list = [150,80,70,50] #Bird A height info stores inside the file names, here we just maunally input it
    else:
        altitude_list = 90 #if meta data is not avaliable, we use 90 meters for all images
    root_dir = '/home/zt253/Models/Retinanet_inference_example/example_images/{}'.format(dataset_folder)
    target_dir = '/home/zt253/Models/Retinanet_inference_example/checkpoint/{}/inference'.format(dataset_folder)
    model_dir = '/home/zt253/Models/Retinanet_inference_example/checkpoint/{}/final_model.pkl'.format(dataset_folder)
    image_out_dir = target_dir
    text_out_dir = target_dir+'/detection-results'
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)
    if (True):
        image_list = sorted(glob.glob(root_dir+'/*.JPG')+glob.glob(root_dir+'/*.jpg')+glob.glob(root_dir+'/*.png'))
        print(image_list)
    else:
        txt_dir = '/media/robert/Backup/UnionData/bird_data_list/test_LBAI.txt'
        with open(txt_dir, 'r') as f:
            txt_data = f.readlines()
        image_list = []
        for data in txt_data:
            image_list.append(data.split(' ')[0])
    inference_mega_image_Retinanet(
		image_list=image_list, model_dir = model_dir, target_dir = target_dir,
		scaleByAltitude=True, defaultAltitude=altitude_list)
