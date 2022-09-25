import pandas as pd
import numpy as np
import os
from PIL  import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 由于有许多标注框是错误的，因此从图片中截取出标注的虫子来过滤
input_csv_path = r'./test/detect_locate修改后.csv'
picture_path = r'./附件1'
output_dir_path = r'./虫子种类的图片（那些本来no_box的）'

isExists = os.path.exists(output_dir_path)
if not isExists:
    os.makedirs(output_dir_path, mode=0o777)

input_table = pd.read_csv(input_csv_path, header=0, encoding='GBK') 
new_table_true = input_table[(input_table['虫子编号'] != 0)]
# print(new_table.head(10))

for index, row in new_table_true.iterrows():
    img_filename = picture_path + '/' + str(row['文件名'])
    img_label = str(row['虫子编号'])
    img_label_name = str(row['虫子名称'])
    
    img_center_x = str(row['中心点x坐标'])
    img_center_y = str(row['中心点y坐标'])
    img_lt_x = int(row['左上角x坐标'])-20
    img_lt_y = int(row['左上角y坐标'])-20
    img_rb_x = int(row['右下角x坐标'])+20
    img_rb_y = int(row['右下角y坐标'])+20
    
    
    insect_own_file = output_dir_path + '/' + img_label + ' ' + img_label_name
    isExists_ = os.path.exists(insect_own_file)
    if not isExists_:
        os.makedirs(insect_own_file, mode=0o777)
    
    # 裁剪图片并保存
    img = Image.open(img_filename)
    crop_box = (img_lt_x, img_lt_y, img_rb_x, img_rb_y)
    img_crop = img.crop(crop_box)
    img_crop.save(insect_own_file +'/' + str(index+1) + '.png')