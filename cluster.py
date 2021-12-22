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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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


def save_labels(path, labels):
    np.savetxt(path, labels)


def save_cluster(path, dic_cluster):
    js_index_cluster = json.dumps(dic_cluster)
    file = open(path, 'w')
    file.write(js_index_cluster)
    file.close()


def load_cluster(path):
    file = open(path, 'r')
    js = file.read()
    try:
        dic = json.loads(js)
    except ValueError:
        print('your file is not a dict file!')
        return {}

    file.close()
    return dic


def get_kmean_clusters(npy_path, class_nums, save_file_path, npy_type='yolov3', show_3d=False):
    '''
    :param npy_path:
    :param class_nums:
    :param save_file_path:
    :param npy_type:
    :show_3d show 2d graph
    :return:
    '''
    if 'txt' in os.path.basename(npy_path):
        pic_list = []
        base_file_list = []
        with open(npy_path, 'r') as file_list:
            lines = file_list.readlines()
            for line in lines:
                value = line.strip()
                dirname, basename = os.path.split(value)
                if not basename[-3:] == 'npy':
                    basename = basename[:-3] + 'npy'
                feat_name = npy_type + '_' + basename
                dirname = dirname.replace('REMAP', 'FEAT')
                dirname = dirname.replace('JPEGImages', 'FEAT')
                feat_path = os.path.join(dirname, feat_name)
                pic_list.append(feat_path)
                base_file_list.append(value)
        data_collection = []
        index_cluster = {}
        issue_files = []
        root_file, _ = os.path.split(npy_path)
        for index, image_file in tqdm(enumerate(pic_list), total=len(pic_list)):
            if not os.path.exists(image_file):
                issue_files.append(base_file_list[index])
                continue
            npy_array = load_npy(image_file)

            pca_array = pca_process(npy_array)
            flat_pca_aray = np.array(pca_array).flatten().tolist()

            data_collection.append(flat_pca_aray)
            # index_cluster[index] = image_file
            index_cluster[index] = base_file_list[index]
        if len(issue_files) > 0:
            issue_files_path = os.path.join(root_file, 'error_feat.txt')
            file = open(issue_files_path, 'w')
            for index, issue_file in enumerate(issue_files):
                file.write(str(issue_file) + '\n')
            file.close()
            print("save not exists feat files in:", issue_files_path)
        if len(data_collection) == 0:
            print('files not exists!')
            return

        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        index_cluster_path = os.path.join(save_file_path, 'index_cluster.txt')
        save_cluster(index_cluster_path, index_cluster)

        array_collection = np.array(data_collection)

        kmeans = KMeans(n_clusters=class_nums, random_state=0).fit(array_collection)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        out_cluster_path = os.path.join(save_file_path, 'out_cluster.txt')
        # save_cluster(out_cluster_path, labels)
        save_labels(out_cluster_path, labels)

        dict_cluster_path = os.path.join(save_file_path, 'dict_filename.txt')
        dict_cluster = {}
        file_cluster_path = os.path.join(save_file_path, 'file_cluster.txt')
        file = open(file_cluster_path, 'w')
        for index, c in enumerate(labels):
            class_index = str(c)
            if not class_index in dict_cluster:
                dict_cluster[class_index] = []
            file_name = index_cluster[index]
            dict_cluster[class_index].append(file_name)
            file.write(class_index + ' ' + str(file_name) + '\n')
        file.close()
        save_cluster(dict_cluster_path, dict_cluster)
        if show_3d:
            # show(array_collection, labels)
            show_tsne(array_collection, labels)

    else:
        print('please input txt file !')


def pca_process(input, n_dims=273):
    '''
    n-dims=273 351
    '''
    n_input = input.reshape(n_dims, -1)
    pca = PCA(n_components=1, svd_solver='arpack')
    pca.fit(n_input)
    X_new = pca.transform(n_input)
    return X_new


def show(data, labels):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(data[:, 3], data[:, 0], data[:, 2], c=labels.astype(float),
               edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title('titles')

    fig.show()


def show_tsne(data, labels):
    import numpy as np
    import sklearn
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits

    # Random state.
    RS = 20150101

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    import matplotlib

    # We import seaborn to make nice plots.
    import seaborn as sns
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})

    digits_proj = TSNE(random_state=RS).fit_transform(data)

    def scatter(x, colors):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 10))

        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                        c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        return f, ax, sc, txts

    scatter(digits_proj, labels)
    plt.savefig('digits_tsne-generated.png', dpi=120)
    plt.show()