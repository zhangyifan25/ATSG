import logging
import datetime
import numpy as np
from texttable import Texttable


def format_tabs1(scores, name_list, cat_list=None):
    _keys = list(scores[0]['iou'].keys())
    _values = []

    for i in range(len(name_list)):
        _values.append(list(scores[i]['iou'].values()))

    _values = np.array(_values) * 100

    t = Texttable()
    t.header(["Class"] + name_list)

    for i in range(len(_keys)):
        t.add_row([cat_list[i]] + list(_values[:, i]))

    t.add_row(["mIoU"] + list(_values.mean(1)))

    return t.draw(),_values.mean(1)

def format_tabs(scores, name_list, cat_list=None):
    
    metric_keys = ['pAcc', 'mAcc', 'miou', 'OA', 'F1']
    _keys = list(scores[0]['iou'].keys())  
    
  
    iou_values = []
    for i in range(len(name_list)):
        iou_values.append(list(scores[i]['iou'].values()))
    iou_values = np.array(iou_values) * 100

    metric_values = []
    for metric in metric_keys:
        metric_row = []
        for i in range(len(name_list)):
            metric_row.append(scores[i][metric] * 100)  
        metric_values.append(metric_row)
    metric_values = np.array(metric_values)

  
    t = Texttable()
   
    t.header(["Metric/Class"] + name_list)


    for i in range(len(_keys)):
        if cat_list and i < len(cat_list):
            class_name = cat_list[i]
        else:
            class_name = f"Class_{_keys[i]}"
        t.add_row([class_name] + [f"{val:.2f}" for val in iou_values[:, i]])

    
    t.add_row(["-" * 10] + ["-" * 8] * len(name_list))

    
    for i, metric in enumerate(metric_keys):
        metric_name = {
            'pAcc': 'Pixel Acc',
            'mAcc': 'Mean Acc', 
            'miou': 'mIoU',
            'OA': 'Overall Acc',
            'F1': 'Mean F1'
        }.get(metric, metric)
        
        t.add_row([metric_name] + [f"{val:.2f}" for val in metric_values[i]])

    return t.draw(), metric_values[metric_keys.index('miou')]


def setup_logger(filename='test.log'):
   
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
  

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v