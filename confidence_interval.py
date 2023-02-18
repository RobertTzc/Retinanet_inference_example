import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import scipy.stats as st

detection_root = '/home/robert/Models/Retinanet_inference_example/results/70/detection-results'
detection_list = glob.glob(detection_root+'/*.txt')
MRE = defaultdict(list)
MSE = defaultdict(list)
for detection_dir in detection_list:
    file_name = os.path.basename(detection_dir)
    gt_dir = '/home/robert/Data/WaterFowl_Processed/drone_collection/'+file_name
    with open(gt_dir,'r') as f:
        gt_data = f.readlines()
    with open(detection_dir,'r') as f:
        data = f.readlines()
    dt_data = []
    for line in data:
        line = line.replace('\n','').split(' ')
        dt_data.append(float(line[1]))
    dt_data = np.asarray(dt_data)
    for confidence in np.arange(2,6,0.5):
        MRE[confidence*0.1].append(min(float('inf'),(len(dt_data[dt_data>confidence*0.1])-len(gt_data))/len(gt_data)))
num_images = len(MRE[0.4])
binwidth = 0.1
plt.title('Model version 120m altitude')
plt.xlabel('relative error')
plt.ylabel('number of images')
re = defaultdict(list)
for idx,confidence in enumerate(np.arange(4,5,0.5)):
    data = np.asarray(MRE[0.1*confidence])
    for i in [0.95,0.9,0.8,0.7]:
        conf_int = st.norm.interval(alpha=i, loc=np.mean(data), scale=st.sem(data))
        conf_int = (round(conf_int[0],4)*100,round(conf_int[1],4)*100)
        re[confidence*0.1].append(conf_int)
        print (i,conf_int,confidence*0.1)
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth),alpha = 0.5,label = 'confidence score: '+str(round(float(confidence*0.1),2)))
print (re)
plt.legend()
plt.grid()
plt.show()
