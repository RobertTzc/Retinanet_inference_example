from encoder import DataEncoder,DataEncoder_fusion
import torch
import json
from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image
import cv2
import math
from utils import read_LatLotAlt,get_GSD,filter_slice
import numpy as np
import matplotlib.pyplot as plt
'''
This script is an envolved version from retinanet_inference.py
designed to prove the concept of slice and merging the feature map
Current this script only support Retinanet_KNN model.
'''
model_conf_threshold = {'Bird_drone_KNN':0.2,}
model_extension = {'Bird_drone_KNN':{20:('_alt_15',15),
                        40:('_alt_30',30),
                        75:('_alt_60',60),
                        90:('_alt_90',90)}
                    }

def get_model_conf_threshold (model_type):
    if (model_type in model_conf_threshold):
        return model_conf_threshold[model_type]
    else:
        return 0.3
def get_model_extension(model_type,model_dir,altitude):
    if(model_type in model_extension):
        model_ext = model_extension[model_type]
        for altitude_thresh in model_ext:
            if (altitude_thresh>=altitude):
                ref_altitude = model_ext[altitude_thresh][1]
                model_dir = model_dir.replace('.pkl',model_ext[altitude_thresh][0]+'.pkl')
                return model_dir,ref_altitude
        model_dir = model_dir.replace('.pkl',model_ext[max(model_ext.keys())][0]+'.pkl')
        return model_dir,model_ext[max(model_ext.keys())][1]
    else:
        return model_dir,altitude

class Retinanet_instance():
    def __init__(self,input_transform,model_type,model_dir,device =torch.device('cuda'),load_w_config = True,altitude=15):
        self.transform = input_transform
        self.model_type = model_type
        self.load_w_config = load_w_config
        self.altitude = altitude
        self.model_dir,self.ref_altitude = get_model_extension(model_type,model_dir,altitude)
        self.device = device
        self.conf_threshold = get_model_conf_threshold(model_type)
        self.model = None
        self.encoder = None
        self.load_model()
        self.p3_block = []
        self.p4_block = []
    def p3_forward(self,module, input, output):
        self.p3_block.append(output)
    def p4_forward(self,module, input, output):
        self.p4_block.append(output)
    
    
    def load_model(self):
        print(self.model_dir)
        if (self.load_w_config):
            config_dir = self.model_dir.replace('.pkl','.json')
            with open(config_dir,'r') as f:
                cfg = json.load(f)
            print (cfg['KNN_anchors'])
            from retinanet_fusion import RetinaNet
            self.model = RetinaNet(num_classes=1,num_anchors=len(cfg['KNN_anchors']))
            self.encoder = DataEncoder_fusion(anchor_wh=cfg['KNN_anchors'],device = self.device)
            #self.model.load_state_dict(torch.load(self.model_dir))
        else:
            from retinanet import RetinaNet
            self.model = RetinaNet(num_classes=1)
            self.encoder = DataEncoder(self.device)
        self.model = torch.load(self.model_dir)
        self.model = self.model.module.to(self.device)
        self.model.eval()
        self.model.down_sample_layer.register_forward_hook(self.p3_forward)
        self.model.p4_layer.register_forward_hook(self.p4_forward)
        
        print('check net mode',next(self.model.parameters()).device)

    def inference(self,image_dir,read_GPS = False,debug = True):
        mega_image = cv2.imread(image_dir)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_BGR2RGB)
        if (read_GPS):
            try:
                altitude = read_LatLotAlt(image_dir)['altitude']
                print ('Reading altitude from Meta data of {}'.format(altitude))
            except:
                altitude = self.altitude
                print ('Meta data not available, use default altitude {}'.format(altitude))
        else:
            altitude = self.altitude
            print ('Using default altitude {}'.format(altitude))
        GSD,ref_GSD = get_GSD(altitude,camera_type='Pro2', ref_altitude=self.ref_altitude)
        ratio = 1.0*ref_GSD/GSD
        ratio = 1
        print('Image processing altitude: {} \t Processing scale {}'.format(altitude,ratio))
        size  = int(ratio*512)
        fm_size = 32
        num_rows,num_columns = math.ceil(1.0*mega_image.shape[0]/size),math.ceil(1.0*mega_image.shape[1]/size)
        padding  = np.zeros((num_columns*size,num_columns*size,3),dtype='uint8')
        fm_map = torch.zeros((1,256,num_columns*fm_size,num_columns*fm_size))
        padding[0:mega_image.shape[0],0:mega_image.shape[1],:] = mega_image
        mega_image = padding
        image = mega_image.copy()
        plt.figure()
        plt.imshow(mega_image)
        for i in range(num_columns):
            for j in range(num_columns):
                sub_image = mega_image[i*size:(i+1)*size,j*size:(j+1)*size,:]
                with torch.no_grad():
                    inputs = self.transform(cv2.resize(
                        sub_image, (512, 512), interpolation=cv2.INTER_AREA)).float()
                    inputs = inputs.unsqueeze(0).to(self.device)
                    loc_preds, cls_preds = self.model(inputs)
                    fm = torch.cat([self.p3_block[-1],self.p4_block[-1]],1)
                    fm_map[:,:,i*fm_size:(i+1)*fm_size,j*fm_size:(j+1)*fm_size] = fm
        fm_map = fm_map
        print (fm_map.shape)
        loc_preds = self.model.loc_head(fm_map).contiguous().view(fm_map.size(0),-1,4)
        cls_preds = self.model.cls_head(fm_map).contiguous().view(fm_map.size(0),-1,1)
        print (mega_image.shape,loc_preds.shape,cls_preds.shape)
        plt.figure()
        plt.imshow(fm_map[0,0,:,:])
        self.encoder.fm_size = fm_size*num_columns
        boxes, labels, scores = self.encoder.decode(
                    loc_preds.data.squeeze(), cls_preds.data.squeeze(), input_size = num_columns*512, CLS_THRESH = self.conf_threshold,NMS_THRESH = 0.25)       
        bbox_list = []
        for idx in range(boxes.shape[0]):
            x1, y1, x2, y2 = list(
                boxes[idx].cpu().numpy())  # (x1,y1, x2,y2)
            score = scores.cpu().numpy()[idx]
            #filter boxes that has overlapped
            bbox_list.append([x1,y1,x2,y2,score])
        for box in bbox_list:
            try:
                cv2.putText(image, str(round(box[4], 2)), (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(image, (int(box[0]), int(
                                box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            except:
                print (box,image.shape)
        return image
                
       

if __name__=='__main__':
    import torchvision.transforms as transforms
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
    model = Retinanet_instance(input_transform = transform,model_type = 'Bird_drone_KNN',
                            model_dir = './checkpoint/Bird_drone_KNN/final_model.pkl',
                            device =torch.device('cpu'),load_w_config = True,altitude=15)
    image_dir = '/home/zt253/data/WaterfowlDataset/Bird_I_Test/HarvestedCrop/DJI_0430.jpg'
    re = model.inference(image_dir=image_dir)
    plt.figure()
    plt.imshow(re)
    plt.show()