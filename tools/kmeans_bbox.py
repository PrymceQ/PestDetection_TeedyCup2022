import numpy as np
import pandas as pd


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    # IOU函数
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        # type(boxes) = array
        # shape(boxes) = n x 2   n个原始框，宽+高
        # type(clusters) = array
        # shape(clusters) = k x 2   k个聚类，宽+高
        
        
        n = boxes.shape[0]      # n个box
        k = cluster_number

        # 计算每个box的面积
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))
        # 最后输出 nxk 矩阵，每一行是k=9个相同的元素，表示第n个box的面积

        # 计算每个聚类box的面积
        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])  # 将y轴扩大一倍，x轴扩大n倍
        cluster_area = np.reshape(cluster_area, (n, k))
        # 最后输出 nxk 矩阵，每一行都有一样的k=9个元素，表示k=9个anchor_box的面积


        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)


        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)  # 交集面积等于两个框的最短宽和最短高相乘

        result = inter_area / (box_area + cluster_area - inter_area)
        
        # 输出 nxk 矩阵， 第 ixj 个格子表示，第i个box与第j个anchor_box的IOU值
        return result

    # 看一下聚类的效果（最终选取的k=9个框的效果）
    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)]) # axis=1 按列给出最大IOU，也就是对所取anchor_box能形成的最大IOU求平均
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k)) # 记录距离
        last_nearest = np.zeros((box_number,)) 
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters 随机选择k个boxes作为初始anchor
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        # 给出最后的聚类框
        return clusters

    def result2txt(self, data):
        f = open("C:/Users/AW15957422232/Desktop/示例数据/附件2/yolov5_anchors.txt", 'w')         # 输出文件位置
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            f.write(x_y + '\n')
        f.close()



    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result


    def get_boxes(self): #根据自己的label方式读取box
        f = pd.read_csv(self.filename, header=0, encoding="gbk")
        f[['左上角x坐标', '左上角y坐标']] = f['左上角坐标'].str.split(',',expand=True) 
        f[['右下角x坐标', '右下角y坐标']] = f['右下角坐标'].str.split(',',expand=True) 
        
        dataSet = []
        for index, row in f.iterrows():
            if row['虫子编号'] == 0:
                pass
            else:
                x_min = int(row['左上角x坐标'])
                x_max = int(row['右下角x坐标'])
                y_min = int(row['左上角y坐标'])
                y_max = int(row['右下角y坐标'])
                
                boxes = [x_min, y_min, x_max, y_max]
                # boxes的构成 -> [x_min, y_min, x_max, y_max] = [左上角x, 左上角y, 右下角x, 右下角y]
    
                width = int(boxes[2] - boxes[0])
                height = int(boxes[3] - boxes[1])
                dataSet.append([width,height])

        result = np.array(dataSet)

        return result


    def txt2clusters(self):
        #all_boxes = self.txt2boxes()
        all_boxes = self.get_boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    # 最终聚类的框个数
    cluster_number = 9
    # csv文件位置
    filename = "C:/Users/AW15957422232/Desktop/泰迪杯准备/【官方】示例数据/附件2/图片虫子位置详情表.csv"

    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
    # 输出9个anchor_box的宽高

