import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fpn import FPN101, FPN50
from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, num_classes=20):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        #fms = [fms]#one layer
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


def img_transform(img_in, transform):
    """B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


if __name__ == '__main__':
    import cv2
    import torchvision.transforms as transforms
    #net  = RetinaNet(2)
    model_dir = '/home/robert/Models/Retinanet_inference_example/checkpoint/Bird_drone_KNN/'
    #print (net._modules.keys(),net.fpn._modules)
    #net.load_state_dict(torch.load(model_dir))
    net = torch.load(model_dir+'final_model_alt_15.pkl')
    print (net.state_dict().keys())
    fmap_block = []
    grad_block = []
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())
    def forward_hook(module, input, output):
        fmap_block.append(output)
    net.module.down_sample_layer.register_forward_hook(forward_hook)
    dummy = torch.zeros((1,3,512,512))
    out = net(dummy)
    print (out[0].shape)
    print (fmap_block[0].shape)