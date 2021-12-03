# coding:utf8

import jieba
import numpy as np
import pandas as pd


def loadDataSet():
    first_data = ["你是一个大傻逼",
                  "今天天气真好啊",
                  "你可真是傻狗",
                  "巴啦啦小魔仙，呜呼啦呼，黑魔法变身！",
                  "你是一头猪么",
                  "哈喽莫妮卡",
                  "恭喜你发财",
                  "你可真是个大天才",
                  "傻逼就是你，你就是傻逼",
                  "你简直是条老狗",
                  "废物说的就是你",
                  "你好聪明啊，我好崇拜你"]
    dataset = []
    for i in first_data:
        temp = list(jieba.cut(i, cut_all=True))
        for j in temp:
            if j == '，':
                temp.remove(j)
        dataset.append(temp)
    classvec = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
    return dataset, classvec


dataSet = loadDataSet()

print(dataSet)
