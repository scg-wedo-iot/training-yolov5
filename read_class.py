# import the os module
import os
import numpy as np
from matplotlib import pyplot as plt

# specify the path to the folder
#folder_path = './OCR_V3.v5/train/labels'
folder_path = './augment_V2/labels'

class_bboxes_list = []
class_balance = []

# loop through the files in the folder
for file_name in os.listdir(folder_path):
    # check if the file is a text file
    if file_name.endswith('.txt'):
        class_bboxes_per_image = []
        # open the file and read its contents
        with open(os.path.join(folder_path, file_name), 'r') as file:            
            bboxes = [np.array(line.split(' ')).astype(float) for line in file.readlines()]
            for box in bboxes:            
                class_box =  int(box[0])
                #class_bboxes_per_image.append(class_box)
                class_balance.append(class_box)
                
        class_bboxes_list.append(class_bboxes_per_image)


#Object Count by Image        
class_list = [0,1,2,3,4,5,6,7,8,9]        
#class_dit = {0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{}}
#for number in class_list:        
 #   for class_bboxes  in class_bboxes_list:        
#        count_number_per_image = class_bboxes.count(number)
#        try:
#            class_dit[number][count_number_per_image] =  class_dit[number][count_number_per_image] + 1
#        except:
#            class_dit[number][count_number_per_image] = 1
            
            
            
# plot hist
plt.hist(class_balance,bins = 20, edgecolor='black',align='mid')
plt.xticks(class_list)
plt.show()








