# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:37:45 2022

@author: fhymy
"""
#file to coco json

#part0 read_csv image_file_dir

#part1 annotation vs no annotation

#part2 dataset_train/val  dict {'image':[],'annontation':[],'categories':[]}

#part3.1 image[{'file_name':'xxxx.jpg','height':int,'weight':int,'id':int},...,{}]

#part3.2 annotation[{'segmentation':[[]],'area':int,'iscrowd':0,'image_id':int,
#                    'bbox':[x,y,w,h],'category_id':int,'id':int},..,{}]

#part3.3 categories[{'supercategory':none,'id':cls,'name':str},...,{}]

#part4 to json

import os
import pandas as pd
import json
#import cv2

anno_pd = pd.read_csv(r'D:\tipdm\图片虫子位置详情表.csv',
                       encoding='ANSI')
#image_root = r'D:\tipdm\insects'
image_root = r'D:\tipdm\附件1'
image_list = os.listdir(image_root)

# with open('train_file.txt','w') as f:
#     f.write(str(image_list).lstrip('[').rstrip(']').replace('\'',''))
# f.close()

test_root = r'D:\tipdm\test\images'
test_list = os.listdir(test_root)

anno_ed = anno_pd.loc[anno_pd['虫子编号'] > 0,:]

# with open('test_file.txt','w') as f:
#     f.write(str(test_list).lstrip('[').rstrip(']').replace('\'',''))
# f.close()

#mask = [i in image_list for i in anno_pd['文件名']]

#mask = [i in list(anno_ed['文件名']) for i in image_list]
image_list_ = list(anno_ed['文件名'].unique())
#row_anno = anno_pd.iloc[mask,:]
train_data = {}

im_list = []
ann_list = []
cate_list = []

for i in image_list_:
    #image_path = image_root + '//' + i
    #im = cv2.imread(image_path)
    #size = im.shape
    h = 3648 #size[0]
    w = 5472 #size[1]
    index = int(i.split('.')[0].lstrip('0'))
    if i not in im_list:
        im_list.append({'file_name':i,'height':h,'width':w,'id':index})
        #index += 1
        
train_data['images'] = im_list



cate_dic = {'大螟':6,'二化螟':7,'稻纵卷叶螟':8,'白背飞虱':9,
            '褐飞虱属':10,'地老虎':25,'蝼蛄':41,'粘虫':105,
            '草地螟':110,'甜菜夜蛾':115,'黄足猎蝽':148,
            '八点灰灯蛾':156,'棉铃虫':222,'二点委夜蛾':228,
            '甘蓝夜蛾':235,'蟋蟀':256,'黄毒蛾':280,'稻螟蛉':310,
            '紫条尺蛾':387,'水螟蛾':392,'线委夜蛾':394,
            '甜菜白带野螟':398,'歧角螟':401,'瓜绢野螟':402,
            '豆野螟':430,'石蛾':480,'大黑鳃金龟':485,'干纹冬夜蛾':673}

# mask_image = list(row_anno['文件名'])
# top_x = list(row_anno['左上角x坐标'])
# top_y = list(row_anno['左上角y坐标'])
# bottom_x = list(row_anno['右下角x坐标'])
# bottom_y = list(row_anno['右下角y坐标'])
# cls_id = list(row_anno['虫子编号'])
anno_ed['右下角x坐标'][anno_ed['右下角x坐标']>5472] = 5472
anno_ed['左上角x坐标'][anno_ed['左上角x坐标']>5472] = 5472
anno_ed['中心点x坐标'][anno_ed['中心点x坐标']>5472] = 5472
anno_ed['右下角y坐标'][anno_ed['右下角y坐标']>3648] = 3648
anno_ed['左上角y坐标'][anno_ed['左上角y坐标']>3648] = 3648
anno_ed['中心点y坐标'][anno_ed['中心点y坐标']>3648] = 3648
mask_image = list(anno_ed['文件名'])
top_x = list(anno_ed['左上角x坐标'])
top_y = list(anno_ed['左上角y坐标'])
bottom_x = list(anno_ed['右下角x坐标'])
bottom_y = list(anno_ed['右下角y坐标'])
cls_id = list(anno_ed['虫子编号'])

for i in range(len(anno_ed)):
    im_id = int((mask_image[i]).split('.')[0].lstrip('0'))
    x = top_x[i]
    y = top_y[i]
    w = bottom_x[i]-x
    h = bottom_y[i]-y
    area = w*h
    box = [x,y,w,h]
    c_id = int(cls_id[i])
    ann_list.append({'segmentation':[[]],'area':area,'iscrowd':0,'image_id':im_id,
                     'bbox':[x,y,w,h],'category_id':c_id,'id':i+1})
    
    
train_data['annotations'] = ann_list

for i in list(cate_dic.keys()):
    cate_list.append({'supercategory':'insect','id':cate_dic[i],'name':i})

train_data['categories'] = cate_list

with open(r'D:\tipdm\insects_train.json','w') as f:
    json.dump(train_data,f)
f.close()



    


   
