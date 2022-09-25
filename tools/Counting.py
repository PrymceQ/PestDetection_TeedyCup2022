# -*- coding: utf-8 -*-

import pandas as pd

result_2 = pd.read_csv(r'D:\tipdm\最终改好的802张图片的预测结果(1).csv',encoding='GBK')
num_result = result_2[['文件名','虫子编号']]
image_list = list(num_result['文件名'].unique())

result_num = pd.DataFrame(columns={'文件名':"", '虫子编号':"", '数量':""})
for i in image_list:
    image = num_result[num_result['文件名']==i]
    class_id = list(image['虫子编号'].unique())
    for idx in class_id:
        if idx == '无':
            result_num=pd.concat([result_num,pd.DataFrame({'文件名':[i],'虫子编号':[0],'数量':[0]})],axis=0)
            continue
        result_num=pd.concat([result_num,pd.DataFrame({'文件名':[i],'虫子编号':[idx],'数量':[len(image[image['虫子编号']==idx])]})],axis=0)

result_num.to_csv(r'D:\tipdm\num_out.csv',encoding='GBK',index=False)