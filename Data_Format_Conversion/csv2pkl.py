# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
result = pd.read_csv(r'D:\tipdm\out.csv',encoding='GBK')
file_list = list(result['文件名'].unique())
#result = pd.read_csv(r'D:\tipdm\yolo_result_428.csv',encoding='GBK')
bbox_num = len(result)

pic_num = len(file_list)
cls_num = 28
insects_id = [6,7,8,9,10,25,41,105,
              110,115,148,156,222,228,
              235,256,280,310,387,392,394,
              398,401,402,430,480,485,673]

pkl_result = []

for file_name in file_list:
    data_pic = result.loc[result['文件名'] == file_name,:]
    pic = []
    for cls_index in insects_id:
        cls_data_pic = data_pic.loc[data_pic['虫子编号']==int(cls_index),:]
        if cls_data_pic.shape[0] < 1:
            pic.append(np.empty(shape=(0,5),dtype=np.float32))
        else:
            #yolo
            inform = np.array(cls_data_pic.iloc[:,2:],dtype=np.float32)
            #inform = np.array(cls_data_pic.iloc[:,2:],dtype=np.float32)
            pic.append(inform)
    pkl_result.append(pic)
            
pkl_path = r'D:\tipdm\result_ensemble.pkl'
pickle.dump(pkl_result,open(pkl_path,'wb'))
