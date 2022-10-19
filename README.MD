# Retinanet for Waterfowl detection inference example

This code is an demo only on how to apply retinanet onto the Waterfowl dataset. In this demo contains pretrained model named as Bird_A, Bird_B, Bird_C, Bird_D, Bird_E  avaialble under folder **checkpoint** please download and unzip the checkpoint folder content using link here https://drive.google.com/file/d/1ZXva0sP5h7JiaUmU4mWpb5yrjwr7B1ai/view?usp=sharing and place the downloaded file inside the checkpoint folder.

## Example images

In this demo, some images are included for testing, under folder **example_images** contains different images for each corresponding pretained model by their folder name(Bird_A,Bird_B etc), along with the GT file attatched for later evaluation.

##installation

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
conda activte toch_py3
```
once you have activated the env, make sure you are nder the  

## Scripts to run

Open **inference_image_list.py** modify folder name and change the root_dir/model_dir if necessary, run the code and inference result will be generated where the model is stored.

## Requirements

Using Anaconda to install virtual env
```
python>=3.8
opencv
pytorch>=11.1
matplotlib
pyexiv2
Pillow
WaterFowlTools (pip install -i https://test.pypi.org/simple/ WaterFowlTools)
numpy
```


