# Installation
## Create Python environment
Using python venv
>python -m venv c:\user\venv\train-yolov5

Using conda
>conda create -m train-yolov5

## Activate Environment
Python venv
>source .../venv/train-yolov5/bin/activate  

Conda
>conda activate train-yolov5

## Install Python Requirements
>pip install -r requirements.txt

## Install yolov5
Clone yolov5  
Change directory to folder "training-yolov5"  
>cd .../training-yolov5  
>git clone https://github.com/ultralytics/yolov5

Install yolov5's requirements  
Change directory to folder "yolov5"  
>cd yolov5  
>pip install -r requirements.txt
