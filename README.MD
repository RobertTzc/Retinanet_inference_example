# Retinanet for Waterfowl detection inference example

This code is an demo only on how to apply retinanet onto the Waterfowl dataset. In this demo contains pretrained model named as Bird_A, Bird_B, Bird_C, Bird_D, Bird_E  avaialble under folder **checkpoint** please download and unzip the checkpoint folder content using link here https://drive.google.com/file/d/1ZXva0sP5h7JiaUmU4mWpb5yrjwr7B1ai/view?usp=sharing and place the downloaded file inside the checkpoint folder.

## Example images

In this demo, some images are included for testing, under folder **example_images** contains different images for each corresponding pretained model by their folder name(Bird_A,Bird_B etc), along with the GT file attatched for later evaluation.

##  installation

###Clone the repository
You can either use the cmd window to clone the repo using the following command or you can refer to the link here:https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
```
git clone https://github.com/RobertTzc/Retinanet_inference_example.git
```
Virtual env is recommended to be used in order to install the package, here Anaconda is recommended to be used, link to the Anaconda https://www.anaconda.com/, once you have installed the Anaconda , refer here to create you virtual env https://conda.io/projects/conda/en/latest/user-guide/getting-started.html. It is recommend to create the env along with python 3.8, demo cmd is here:
```
conda create -n torch_py3 python==3.8
```
once env is created use, eg:
```
conda activaate torrch_py3
```
once you have activated the env, make sure you are under the env you created and run the following commands:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install opencv-contrib-python
pip install Pillow==6.1
pip install pandas
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
```
## input format
This script requires a speical format of iunput describes below
```
Image_folder (eg Bird_A)
├── image_name1.jpg
├── image_name1.txt # if want to evaluate
├── image_name2.jpg
├── image_name1.txt # if want to evaluate
├── image_name3.jpg
├── image_name2.txt # if want to evaluate
├── ...
└── image_info.csv
```
Inside the image_info.csv we requires at least the following columns:
```
image_name    |     Date     |   location   |   latitude   |   longitude   |   altitude    |
------------- | -------------| -------------| -------------| ------------- | ------------- | 
2_80_3.jpg    |  2022_10_17  |  abc         |  N/A         |  N/A          |  80           |
<required>    |  <optional>  |  <optional>  |  <optional>  |  <optional>   |  <required>   |
```
required are values critical to the program and optionals are description data for the record, if you have empty values, simply put N/A on optional values.

## Run the Scripts:
Once you have the input file ready and in correct virtual env, you can use the file **inference_image_list.py** to start inference the images:
quick example:
```
python inference_image_list.py --model_dir ./checkpoint/Bird_A/final_model.pkl --model_type Bird_A --image_root ./example_images/Bird_A --csv_dir ./example_images/Bird_A/image_info.csv --evaluate true --ext JPG --visualize true
```
Avilable commands are:
```
--model_dir: directory of the model, in this demo is where the checkpoint model locates
--model_type: we include some models that are specifically trained on different types of scenerio, in this demo for most cases, it will --simply match the model_dir type.
--image_root: specify where the iunput images stores
--csv_dir: a csv file specifiy where the image info stores, usually was same place with the image_root dir.
--out_dir: where the output file will be generated, by default it will create 'Result' folder under current directory.
--ext: specify what the extension of the file will be, default is JPG, common ext are jpg,png,JPG
--visual: True/False value specify whether we want to have visualization on output, default is True
--evaluate: whether we want to evaluate the result, this can only be done when the input file comes with groundTruth file, default is False
```
## Output format
When you specify the output dir, you shall expecting the output in the following:
```
Result folder 
├── detection-results
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── visualize-results
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
├── configs.json
├── detection_summary.csv
├── f1_score.jpg    #if apply evaluation
└── mAP.jpg         #if apply evaluation

```
