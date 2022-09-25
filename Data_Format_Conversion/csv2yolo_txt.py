import pandas as pd
import numpy as np
from PIL import Image
import os
import csv

# 进行归一化操作
def convert(size, box):
    # size指图片的宽高的元组
    # box指框的x轴最左、x轴最右、y轴最下、y轴最上的元组
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def td_csv2labels(home_path):
    input_csv_path = home_path + r'附件2/图片虫子位置详情表.csv'
    output_dir_path = home_path + r'labels'
    picture_path = home_path + r'附件1'
    
    # mkdir labels
    isExists = os.path.exists(output_dir_path)
    if not isExists:
        os.makedirs(output_dir_path, mode=0o777)

    # 读取csv
    input_table = pd.read_csv(input_csv_path, header=0, encoding="gbk")
    input_table[['左上角x坐标', '左上角y坐标']] = input_table['左上角坐标'].str.split(',',expand=True) 
    input_table[['右下角x坐标', '右下角y坐标']] = input_table['右下角坐标'].str.split(',',expand=True) 
    
    # 用于计数
    count_all = 0  #计数所有非0类别
    count_each_class = [0 for i in range(29)] # 记录29各类每个类数量（包括0）

    # 对csv行进行遍历
    for index, row in input_table.iterrows():
        img_ID = str(row['文件名']).split('.')[0]
        img_label = str(row['虫子编号'])
        img_label_name = str(row['虫子名称'])
        
        assert(classes_chn.index(img_label_name) == classes.index(img_label)) #判断编号和名称是否对应
        
        # 建立label文件夹下的txt进行box记录
        txt = img_ID + '.txt'
        out_file = open(output_dir_path + '/' + txt, 'a', encoding='UTF-8') 
        
        if img_label == str('0'):
            # 只建立相对应的空白txt
            pass
        else:
            img_path = picture_path + '/' + img_ID + '.jpg'
            img = Image.open(img_path)

            # 将信息写入txt
            h = img.height #原图height
            w = img.width
            x_min = row['左上角x坐标']
            x_max = row['右下角x坐标']
            y_min = row['左上角y坐标']
            y_max = row['右下角y坐标']

            # x_min = row['x_min']
            # x_max = row['x_max']
            # y_min = row['y_min']
            # y_max = row['y_max']


            box = convert((int(w), int(h)), (int(x_min), int(x_max), int(y_min), int(y_max)))
            
            label = classes.index(img_label)
            out_file.write(str(label) + ' ' + ' '.join([str(round(a, 6)) for a in box]) + '\n')
            
            # 类别计数
            count_all += 1

        index = classes.index(img_label)
        count_each_class[index] = count_each_class[index] + 1


    # 将计数内容建立表格进行展示
    DF_display = pd.DataFrame({'index': range(29),'名称': list(classes_chn),'编号': list(classes), '计数':list(count_each_class)})
    
    # 输出dataframe对齐
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 180) # 设置打印宽度(**重要**)
    
    print(DF_display)
    print("害虫总数量：", count_all)
    
    
if __name__ == '__main__':
    home_path = r'C:/Users/wz/Desktop/datasets/'
    
    # 29个类的中文序列
    classes_chn = ['无', '大螟', '二化螟', '稻纵卷叶螟', '白背飞虱', '褐飞虱属',
            '地老虎', '蝼蛄', '粘虫', '草地螟', '甜菜夜蛾', '黄足猎蝽', '八点灰灯蛾',
            '棉铃虫', '二点委夜蛾', '甘蓝夜蛾', '蟋蟀', '黄毒蛾', '稻螟蛉', '紫条尺蛾',
            '水螟蛾', '线委夜蛾', '甜菜白带野螟', '歧角螟', '瓜绢野螟', '豆野螟', 
            '石蛾', '大黑鳃金龟', '干纹冬夜蛾']
    # 29个类的编号序列
    classes = ['0', '6', '7', '8', '9', '10', '25', '41', '105', '110', '115', '148',
            '156', '222', '228', '235', '256', '280', '310', '387', '392', '394', 
            '398', '401', '402', '430', '480', '485', '673']
    
    labels_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
              '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
              '20', '21', '22', '23', '24', '25', '26', '27', '28']
    
    td_csv2labels(home_path=home_path)