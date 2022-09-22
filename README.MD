# Retinanet for Waterfowl detection inference example

This code is an demo of how to apply retinanet onto the Waterfowl dataset, demo contains pretrained model on Bird_A, Bird_B, Bird_C, Bird_D, Bird_E under folder **checkpoint** please download and unzip the checkpoint folder using link here https://drive.google.com/file/d/1ZXva0sP5h7JiaUmU4mWpb5yrjwr7B1ai/view?usp=sharing

## Example images

Images are included for testing, under folder **example_images** with each pretained model, along with the GT file attatched for instant evaluation.

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


