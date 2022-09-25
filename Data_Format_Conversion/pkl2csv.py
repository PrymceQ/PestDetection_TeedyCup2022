# -*- coding: utf-8 -*-

import csv
import pandas as pd
import pickle
import os
#import numpy as np
#import pprint
file=open(r"D:\tipdm\final_rcnn.pkl","rb")
data=pickle.load(file)
#pprint.pprint(data)
file.close()
#gt_df = pd.read_csv(r'D:\tipdm\图片虫子位置详情表420.csv',encoding='ANSI')

#pikle list[picture1,...picture_n]
#picture_i list[class1,...class_m]
#class_j np.array[bbox1,...,bbox_] k*5 
#bbox 1*5 x1,y1,x2,y2,conf

#file_name list <-> picture_i

#image_root = r'D:\tipdm\images'
#image_list = os.listdir(image_root)

#anno_pd = pd.read_csv(r'D:\tipdm\图片虫子位置详情表.csv',
#                       encoding='ANSI')
#image_list = list(anno_pd['文件名'].unique())
#image_root = r'D:\tipdm\insects'
#image_root = r'D:\tipdm\附件1'
#image_list = os.listdir(image_root)

# with open('train_file.txt','w') as f:
#     f.write(str(image_list).lstrip('[').rstrip(']').replace('\'',''))
# f.close()

test_root = r'D:\tipdm\测试数据\测试图像文件'
image_list = os.listdir(test_root)

#anno_ed = anno_pd.loc[anno_pd['虫子编号'] > 0,:]

# with open('test_file.txt','w') as f:
#     f.write(str(test_list).lstrip('[').rstrip(']').replace('\'',''))
# f.close()

#mask = [i in image_list for i in anno_pd['文件名']]

#mask = [i in list(anno_ed['文件名']) for i in image_list]
#image_list_ = list(set(anno_ed['文件名']))


bbox = []

insects = {1:'person'}

for i in range(len(data)):
    picture_i = data[i]
    class_num = len(picture_i)
    file_name = image_list[i]
    have_insects = False
    for j in range(class_num):
        insects_id = insects[j+1]
        class_j = picture_i[j]
        bbox_num = class_j.shape[0]
        if bbox_num > 0:
            for k in range(bbox_num):
                bbox_class_j = class_j[k]
                conf = bbox_class_j[4]
                if conf < 0.2: 
                    continue
                have_insects = True
                
                tl_x = int(bbox_class_j[0])
                tl_y = int(bbox_class_j[1])
                br_x = int(bbox_class_j[2])
                br_y = int(bbox_class_j[3])
                
                c_x = int((tl_x+br_x)/2)
                c_y = int((tl_y+br_y)/2)
                bbox.append([file_name,insects_id,c_x,c_y,tl_x,tl_y,br_x,br_y,conf])
        else:
            continue
    if not have_insects:
        bbox.append([file_name,'无','','','','','','',''])
    

save_dir = r"D:\tipdm\final_rcnn.csv"

with open(save_dir,"w",newline='') as csvfile: 
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(["文件名","虫子编号","中心点x坐标","中心点y坐标","左上角x坐标",
                     "左上角y坐标","右下角x坐标","右下角y坐标","conf"])
    #写入多行用writerows
    writer.writerows(bbox)



    



    