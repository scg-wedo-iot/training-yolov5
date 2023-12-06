#https://github.com/facebookresearch/nougat/issues/40
#test
import albumentations as A
import cv2
import numpy as np
import os
import shutil
import sys

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

# train/images
# train/labels
folder_data_img_label = os.path.join(FOLDER_PROJECT, 'data', 'raw', 'test_train_split', 'train')
folder_images_path = os.path.join(folder_data_img_label, 'images')
folder_bboxes_path = os.path.join(folder_data_img_label, 'labels')

folder_data_augment = os.path.join(FOLDER_PROJECT, 'data', 'raw', 'test_train_split_aug', 'train')
save_img_path = os.path.join(folder_data_augment, 'images')
if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
save_label_path = os.path.join(folder_data_augment, 'labels')
if not os.path.exists(save_label_path):
    os.makedirs(save_label_path)

def move_1st_to_end(lst):
  #moving to end
  lst = [x for x in lst if x != lst[0]] + [int(lst[0])]
  
  return lst

def move_to_end_list(list_box):
    new_boxes = []
    
    for bbox in list_box:
        new_boxes.append(move_1st_to_end(bbox))
    
    return new_boxes 

def txt_to_bboxesformat(bboxes_path): 
    class_bboxes_per_image = []
    #read bboxes_path
    bboxes_read = open(bboxes_path, 'r')
    # Set format
    bboxes = [np.array(line.split(' ')).astype(float) for line in bboxes_read.readlines()]
    
    for box in bboxes:            
        class_box =  int(box[0])
        class_bboxes_per_image.append(class_box)
    
    count_number_per_image = class_bboxes_per_image.count(0)  
    bboxes = move_to_end_list(bboxes)
    
    bboxes_read.close()
      
    return bboxes    

def augmentation(image,bboxes,save_img_path,save_label_path,nameNewImage,nameNewLabel,num_time = 1):
    height, width, _ = image.shape
    
    #Gen Augment
    for count in range(num_time):
        transform = A.Compose([
             A.ToGray(),
             A.HueSaturationValue(),
             A.RandomBrightnessContrast(),
             A.Posterize(),
             A.Defocus(always_apply=False, p=0.5, radius=(1, 4), alias_blur=(0.0, 0.3)),
             A.RGBShift(always_apply=False, p=0.5, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
             A.Rotate(always_apply=False, p=0.5, limit=(-5, 5), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False)
           ], bbox_params=A.BboxParams(format='yolo',min_visibility =0.3))
    
        # Augment an image
        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        
        # Save png
        path_save_img = os.path.join(save_img_path, str(count) + "_" + nameNewImage)
        cv2.imwrite(path_save_img, transformed_image)
    
        # Save txt
        path_save_txt = os.path.join(save_label_path, str(count) + "_" + nameNewLabel)
        with open(path_save_txt, 'w') as file:
            for bbox in transformed_bboxes:
               labels = str(bbox[-1])+" "+ " ".join(map(str, bbox[:-1]))
               file.write(labels)    
               file.write("\n")

count = 0 

for file_name in os.listdir(folder_images_path):
    bboxes_name = file_name[:-4] + ".txt"
    
    image_path = os.path.join(folder_images_path, file_name)   
    bboxes_path = os.path.join(folder_bboxes_path, bboxes_name)
    
    # Read an image with OpenCV
    #image_path= "./OCR_V3.v5/train/images/17_jpg.rf.57cc621f786fb1a94183c250201adffe.jpg"
    #bboxes_path = "./OCR_V3.v5/train/labels/17_jpg.rf.57cc621f786fb1a94183c250201adffe.txt"
    
    bboxes = txt_to_bboxesformat(bboxes_path)
    image = cv2.imread(image_path)

    # Save origin
    shutil.copy2(image_path, os.path.join(save_img_path, file_name))
    shutil.copy2(bboxes_path,os.path.join(save_label_path, bboxes_name))

    augmentation(image,bboxes,
                save_img_path = save_img_path,save_label_path = save_label_path,
                nameNewImage = file_name ,nameNewLabel = bboxes_name,
                num_time = 8)

    count = count + 1
    
    print(int(count*100/len(os.listdir(folder_images_path))))

    
    