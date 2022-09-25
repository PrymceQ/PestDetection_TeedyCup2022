# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'runs/train/coco128/weights/best.pt',  # model.pt path(s) 
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam            
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image 
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size(32x32)

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0   ## dt用来记录3个过程的时间：1、读取图片 2、图片推理时间 3、NMS后处理时间
    
    
    txt_path_count = str(save_dir / 'count_sum')                                                                            #（改）用来计数每张图片中每个类别出现的数量
    with open(txt_path_count + '.txt', 'a') as f:                                                                           # 变量有：txt_path_count s_for_count s_for_count1
            f.write('picture_ID ' + 'image_size ' + 'object_ID ' + 'obejct_name ' + 'object_sum' + '\n')
            
    txt_path_locate = str(save_dir / 'detect_locate')                                                                       #（改）用来计数每张图片中每个类别出现的位置信息
    with open(txt_path_locate + '.txt', 'a') as f:                                                                          # 变量有：txt_path_locate 
            f.write('picture_ID ' + 'image_size ' + 'object_ID ' + 'obejct_name ' + 'center_x ' + 'center_y ' + 'topleft_x ' + 'topleft_y ' + 'lowerright_x ' + 'lowerright_y '+ 'conf'+ '\n')
    
    for path, im, im0s, vid_cap, s in dataset:
        
        s_for_count = s.split('\\')[-1]
        s_for_count = s_for_count[:-2] + " "
    
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image   det是检测出来框的集合
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            s_for_count += '%gx%g ' % im.shape[2:]
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    s_for_count1 = s_for_count
                    s_for_count1 += f"{int(c)} {names[int(c)]} {n}" 
                    with open(txt_path_count + '.txt', 'a') as f:                                                    
                        f.write(s_for_count1+ '\n')

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        
                        
                        x1y1x2y2 = s_for_count     #记录左上角坐标和右下角坐标
                        #左上角坐标
                        x1=int(xyxy[0].item())
                        y1=int(xyxy[1].item())
                        #右下角坐标
                        x2=int(xyxy[2].item())
                        y2=int(xyxy[3].item())
                        #中心点坐标
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        x1y1x2y2 += f"{int(c)} {names[int(c)]} {x_center} {y_center} {x1} {y1} {x2} {y2} {conf}" 
                        
                        #打印
                        with open(txt_path_locate + '.txt', 'a') as f:                                                    
                            f.write(x1y1x2y2 + '\n')


            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

# 检测参数配置
def parse_opt():
    parser = argparse.ArgumentParser()
    
    ### 必要文件调取
    # 选用训练的权重，不指定的话会使用yolov5l.pt预训练权重
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/8888/weights/best.pt', help='model path(s)')
    # 检测数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头)，也可以是rtsp等视频流
    parser.add_argument('--source', type=str, default='/home/detections/teddy_cuo_data/附件3_802张预测图', help='file/dir/URL/glob, 0 for webcam')
    # 指定推理图片分辨率，默认640（32倍数），先将图片转成640x640，保存的时候在还原为原尺寸
    parser.add_argument('--data', type=str, default=ROOT / 'data/insect.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    
    ### 模型超参数设置
    # 置信度阈值，检测到的对象属于特定类（狗，猫，香蕉，汽车等）的概率，默认为0.25
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # 指定NMS(非极大值抑制)的IOU阈值，默认为0.45
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # 每张图最多检测多少目标，默认为1000个
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # 检测的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)。值为空时，训练时默认使用计算机自带的显卡或CPU
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    ### 保存检测效果设置
    # 是否展示检测之后的图片/视频，默认False
    parser.add_argument('--view-img', default = False, action='store_true', help='show results')
    # 是否将检测的框坐标以txt文件形式保存(yolo格式)，默认False
    parser.add_argument('--save-txt', default = True, action='store_true', help='save results to *.txt')
    # 在输出标签结果txt中同样写入每个目标的置信度，默认False
    parser.add_argument('--save-conf', default = False, action='store_true', help='save confidences in --save-txt labels')
    # 从图片\视频上把检测到的目标抠出来保存，默认False
    parser.add_argument('--save-crop', default = False, action='store_true', help='save cropped prediction boxes')
    # 不保存图片/视频，默认False
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    
   # 设置只检测特定的类，如--classes 0 2 4 6 8，默认False
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # 使用agnostic NMS(前背景)，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 推理的时候进行多尺度，翻转等操作(TTA)推理，属于增强识别，速度会慢不少，默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 特征可视化，默认False
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    
    # 更新所有模型，默认False
    parser.add_argument('--update', action='store_true', help='update all models')
    # 检测结果所存放的路径，默认为runs/detect
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    # 检测结果所在文件夹的名称，默认为exp
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 若现有的project/name存在，则不进行递增
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 画图时线条宽度
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # 隐藏标签
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # 隐藏置信度
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # 半精度检测(FP16)
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 在onnx推理中使用OpenCV DNN
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
