import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from scipy import ndimage


#Create function to crop images.
def crop(img, bg, mask) -> np.array:
    '''
    Function takes image, background, and mask, and crops the image.
    The cropped image should correspond only with the positive portion of the mask.
    '''
    fg = cv2.bitwise_or(img, img, mask=mask) 
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image

def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def rotationAng(cnt,image):
    
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    if angle > 80:
        return image
    
    #rotation angle in degree
    rotated = ndimage.rotate(image, angle)
      
    return rotated

import pandas as pd
import ast

#df = pd.read_csv("2023-03-28T010053_68B6B34E8D4C_Chon_Buri_true.csv")
folderImg = "WHA_demo/Lab_SCG/"
folderSaveImg = "images_test/"
#folderSaveTxt = "datasets/water_meter/labels/"
img_list = os.listdir(folderImg)

for img_name in img_list:
#for index in range(len(df)):
    try:
        #print(int(index/len(df)))
        #device_info = df["device"].loc[index]
        #date,time = df["timestamp"].loc[index].split()
        #ocr_ref = df["ocr_ref"].loc[index]
        #time = time.split(':') 
        
        #device_info_json = ast.literal_eval(device_info)
        #img_name = device_info_json["mac_id"] + "_"+ date +"_"+''.join(time) +".png"
         
        #path add
        img_path_full = folderImg + img_name   
        print(img_name)
        #Open Image
        raw = cv2.imread(img_path_full)
        #plt.imshow(raw)
        #plt.show()
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw_copy = raw
        y_raw_copy,x_raw_copy,_ = raw_copy.shape 
        raw = cv2.resize(raw, (256, 256), interpolation = cv2.INTER_AREA)
        raw = np.array(raw)/255.
        
        #predict the mask 
        model = load_model('unet_V2.h5',compile = False)
        pred = model.predict(np.expand_dims(raw, 0))
        
        #mask post-processing 
        msk  = pred.squeeze()
        msk = np.stack((msk,)*3, axis=-1)
        msk[msk >= 0.5] = 1 
        msk[msk < 0.5] = 0 
        
        #dilate mask
        kernel = np.ones((5,5),np.uint8)
        msk = cv2.dilate(msk,kernel,iterations = 1)
        
        msk_resize = cv2.resize(msk, (x_raw_copy, y_raw_copy), interpolation = cv2.INTER_AREA)
        
        #crop img 
        crop_img = raw_copy * msk_resize
        crop_img = crop_img.astype(np.uint8)
        
        #show the mask and the segmented image 
        combined = np.concatenate([raw_copy, msk_resize,crop_img], axis = 1)
        
        #find crop img from mask
        white = np.where(msk_resize != 0)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        roi_img = crop_img[ymin+5:ymax+5, xmin:xmax]
        raw_roi_img = raw_copy[ymin+5:ymax+5, xmin:xmax]
        #plt.imshow(raw_roi_img)
        #plt.show()
        
        #rescale [0..1] -> [0..255]
        roi_img_test = roi_img * 255
        
        #find contours
        roi_img_test = roi_img_test.astype(np.uint8)   
        roi_img_test = cv2.cvtColor(roi_img_test, cv2.COLOR_BGR2GRAY)         
        _,thresh = cv2.threshold(roi_img_test,0,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cnt = contours[0]
        #rect  = cv2.minAreaRect(cnt)
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #raw_segment = cv2.drawContours(roi_img,[box],0,(0,0,255),2)
        
        img_rota = rotationAng(cnt,raw_roi_img)
        
        y,x,_ = img_rota.shape
        plt.imshow(img_rota)
        plt.show()
        count = 0
        step = int(x/6 *0.8)
        y_star = int(0.2*y)
        y_end = int(0.9*y)   
        x_star = int(0.05*x)        
        x_end = int(0.9*x) 
        
        nameSaveImg = folderSaveImg +"ROI_"+ img_name.split(".")[0] +".jpg"
        #nameSaveTxt = folderSaveTxt +"ROI_"+ img_name.split(".")[0] + ".txt"
        #class_nuber = [int(x) for x in str(ocr_ref)]
        
        img_rota = cv2.cvtColor(img_rota, cv2.COLOR_BGR2RGB)
        cv2.imwrite(nameSaveImg, img_rota)
        
        #f = open(nameSaveTxt, "w")
        
        for x_index in range(x_star,x_end,step):
        
            if count < 7:
                number = img_rota[y_star:y_end,x_index-(int(step*0.2)):x_index+step]
                plt.imshow(number)
                plt.show()
           

                
                x_center = (((x_index+step) + (x_index-(int(step*0.2))) ) /2) /x 
                y_center = ((y_end + y_star)/2)/y     
                width  =  ((x_index+step) - (x_index-(int(step*0.2)))) / x
                height = (y_end - y_star) / y
                
               
              
                
                #plt.imshow(img_rota)
                #plt.scatter(int(mid_x*x),int(mid_y*y))
                #plt.show()
                
                #f.write(f"{class_nuber[count]} {x_center} {y_center} {width} {height}\n")
                count = count + 1    
                
        #f.close()
            
           
          
        
        
        #class x_center y_center width height
        
        
        #plt.axis('off')
        #plt.imshow(crop_img)
        #plt.show()
        #plt.imshow(roi_img)
        #plt.show()
        #plt.imshow(img_rota)
        #plt.show()
        #plt.imshow(raw_segment)
        #plt.show()
        #plt.imshow(crop_img)
        #plt.scatter(box[1][0],box[1][1])
        #plt.show()
        
    except:        
        print("Error")
    




