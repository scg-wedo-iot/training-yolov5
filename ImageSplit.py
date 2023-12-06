#test
import os
from sklearn.model_selection import train_test_split
import shutil

import math
import sys
import os
import cv2
import pandas as pd
import time
import numpy as np

def checkSysPathAndAppend(path, stepBack=0):
    if stepBack > 0:
        for istep in range(stepBack):
            if istep == 0:
                pathStepBack = path
            pathStepBack, filename = os.path.split(pathStepBack)
    else:
        pathStepBack = path

    if not pathStepBack in sys.path:
        sys.path.append(pathStepBack)

    return pathStepBack

folderFile, filename = os.path.split(os.path.realpath(__file__))
FOLDER_PROJECT = checkSysPathAndAppend(folderFile, 2)


def saveData(data,folder_input,folder_output):
    for name in data:
        image_path = os.path.join(folder_input, name)
        if not os.path.exists(folder_output):
            os.makedirs(folder_output)
        save_path = os.path.join(folder_output, name) 
        shutil.copy2(image_path,save_path)

def makeBboxes_list(images_list):
    bboxes_list = []
    
    for file_name in images_list:
        bboxes_name = file_name[:-4] + ".txt"
        bboxes_list.append(bboxes_name)
        
    return bboxes_list

# --------- Input ----------
folder_images = os.path.join(FOLDER_PROJECT, 'data', 'raw', 'test_train')
folder_images_path = os.path.join(folder_images, "images")
folder_bboxes_path  = os.path.join(folder_images, "labels")

folder_images_split = os.path.join(FOLDER_PROJECT, 'data', 'raw', 'test_train_split')
folder_train_images_path = os.path.join(folder_images_split, "train", "images")
folder_train_bboxes_path = os.path.join(folder_images_split, "train", "labels")

folder_val_images_path = os.path.join(folder_images_split, "val", "images")
folder_val_bboxes_path = os.path.join(folder_images_split, "val", "labels")

images_list = os.listdir(folder_images_path)
bboxes_list = makeBboxes_list(images_list)

img_train, img_val, bboxes_train, bboxes_val = train_test_split(images_list, bboxes_list, test_size=0.2)

# train img
print("split training data...")
saveData(img_train,folder_images_path,folder_train_images_path)
# train bbox
saveData(bboxes_train,folder_bboxes_path,folder_train_bboxes_path)
# val img
print("split validation data...")
saveData(img_val,folder_images_path,folder_val_images_path)
# val bbox
saveData(bboxes_val,folder_bboxes_path,folder_val_bboxes_path)

print("----------------------")
print(f"img_train : {len(img_train)}")
print(f"bboxes_train : {len(bboxes_train)}")
print(f"img_test : {len(img_val)}")
print(f"bboxes_test : {len(bboxes_val)}")
