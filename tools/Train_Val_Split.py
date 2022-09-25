import pandas as pd
import numpy as np
import os
import csv
import shutil

input_csv_path = r'./附件2/图片虫子位置详情表.csv'
picture_path = r'./train/images/附件1'
input_csv_path2 = r'./附件3/无位置信息的图片汇总表.csv'

# input_table = pd.read_csv(input_csv_path, header=0, encoding='GBK')
input_table2 = pd.read_csv(input_csv_path2, header=0, encoding='GBK')

# new_table_true = input_table[(input_table['虫子编号'] != 0)]
# new_table_false = input_table[(input_table['虫子编号'] == 0)]
# print(new_table.head(10))

for index, row in input_table2.iterrows():
    img_ID = str(row['文件名'])
    shutil.copyfile('./附件1/' + str(img_ID), './附件3_802张预测图/' + str(img_ID))


# for index, row in new_table_false.iterrows():
#     img_ID = str(row['文件名'])
#     shutil.copyfile('./附件1/' + str(img_ID), './test/images/' + str(img_ID))
    
















# if __name__ == '__main__':
#     home_path = r'E:/泰迪杯数据/正式数据/2022.04.06(正式数据)/'
    
#     # 29个类的中文序列
#     classes_chn = ['无', '大螟', '二化螟', '稻纵卷叶螟', '白背飞虱', '褐飞虱属',
#             '地老虎', '蝼蛄', '粘虫', '草地螟', '甜菜夜蛾', '黄足猎蝽', '八点灰灯蛾',
#             '棉铃虫', '二点委夜蛾', '甘蓝夜蛾', '蟋蟀', '黄毒蛾', '稻螟蛉', '紫条尺蛾',
#             '水螟蛾', '线委夜蛾', '甜菜白带野螟', '歧角螟', '瓜绢野螟', '豆野螟', 
#             '石蛾', '大黑鳃金龟', '干纹冬夜蛾']
#     # 29个类的编号序列
#     classes = ['0', '6', '7', '8', '9', '10', '25', '41', '105', '110', '115', '148',
#             '156', '222', '228', '235', '256', '280', '310', '387', '392', '394', 
#             '398', '401', '402', '430', '480', '485', '673']
    
#     labels_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
#               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
#               '20', '21', '22', '23', '24', '25', '26', '27', '28']
    
#     td_csv2labels(home_path=home_path)