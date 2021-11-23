# !/usr/bin/python
# coding: utf8
# @Time    : 2021-11-22
# @Author  : LYK
# @Software: PyCharm

from collections import defaultdict
import numpy as np
import copy
from resnet_feat_extract import load_npy
import os
from tqdm import tqdm
import json

class KMEANS:
    def __init__(self, n_cluster, epsilon=1e-3, maxstep=2000):
        self.n_cluster = n_cluster
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.N = None
        self.centers = None
        self.cluster = defaultdict(list)

    def init_param(self, data):
        # 初始化参数, 包括初始化簇中心
        self.N = data.shape[0]
        random_ind = np.random.choice(self.N, size=self.n_cluster)
        self.centers = [data[i] for i in random_ind]  # list存储中心点坐标数组
        for ind, p in enumerate(data):
            self.cluster[self.mark(p)].append(ind)
        return

    def _cal_dist(self, center, p):
        # 计算点到簇中心的距离平方
        return sum([(i - j) ** 2 for i, j in zip(center, p)])

    def mark(self, p):
        # 计算样本点到每个簇中心的距离，选取最小的簇
        dists = []
        for center in self.centers:
            dists.append(self._cal_dist(center, p))
        return dists.index(min(dists))

    def update_center(self, data):
        # 更新簇的中心坐标
        for label, inds in self.cluster.items():
            self.centers[label] = np.mean(data[inds], axis=0)
        return

    def divide(self, data):
        # 重新对样本聚类
        tmp_cluster = copy.deepcopy(self.cluster)  # 迭代过程中，字典长度不能发生改变，故deepcopy
        for label, inds in tmp_cluster.items():
            for i in inds:
                new_label = self.mark(data[i])
                if new_label == label:  # 若类标记不变，跳过
                    continue
                else:
                    self.cluster[label].remove(i)
                    self.cluster[new_label].append(i)
        return

    def cal_err(self, data):
        # 计算MSE
        mse = 0
        for label, inds in self.cluster.items():
            partial_data = data[inds]
            for p in partial_data:
                mse += self._cal_dist(self.centers[label], p)
        return mse / self.N

    def fit(self, data):
        self.init_param(data)
        step = 0
        while step < self.maxstep:
            step += 1
            self.update_center(data)
            self.divide(data)
            err = self.cal_err(data)
            if err < self.epsilon:
                break
        return


def save_cluster(path, dic_cluster):
    js_index_cluster = json.dumps(dic_cluster)
    file = open(path, 'w')
    file.write(js_index_cluster)
    file.close()


def load_cluster(path):
    file = open(path, 'r')
    js = file.read()
    dic = json.loads(js)
    file.close()
    return dic


def get_kmean_clusters(npy_path, class_nums, save_file_path, npy_type='yolov3'):
    '''

    :param npy_path:
    :param class_nums:
    :param save_file_path:
    :param npy_type:
    :return:
    '''
    if 'txt' in os.path.basename(npy_path):
        pic_list = []
        with open(npy_path, 'r') as file_list:
            lines = file_list.readlines()
            for line in lines:
                value = line.strip()
                dirname, basename = os.path.split(value)
                if not basename[-3:] == 'npy':
                    basename = basename[:-3] + 'npy'
                feat_name = npy_type + '_' + basename
                feat_path = os.path.join(dirname, feat_name)
                pic_list.append(feat_path)
        data_collection = []
        index_cluster = {}
        for index, image_file in tqdm(enumerate(pic_list), total=len(pic_list)):
            if not os.path.exists(image_file):
                continue
            npy_array = load_npy(image_file)
            flat_npy_aray = np.array(npy_array).flatten().tolist()

            data_collection.append(flat_npy_aray)
            index_cluster[index] = image_file
        if len(data_collection) == 0:
            print('files not exists!')
            return

        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        index_cluster_path = os.path.join(save_file_path, 'index_cluster.txt')
        save_cluster(index_cluster_path, index_cluster)

        array_collection = np.array(data_collection)
        km = KMEANS(class_nums)
        km.fit(array_collection)
        cluster = km.cluster
        # centers = np.array(km.centers)
        print(cluster)

        out_cluster_path = os.path.join(save_file_path, 'out_cluster.txt')
        save_cluster(out_cluster_path, cluster)

        file_cluster_path = os.path.join(save_file_path, 'file_cluster.txt')
        file = open(file_cluster_path, 'w')
        for c in cluster:
            index_list = cluster[c]
            for index in index_list:
                file_name = index_cluster[index]
                file.write(str(c) + ' ' + str(file_name) + '\n')
        file.close()

    else:
        print('please input txt file !')
